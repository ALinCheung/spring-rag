package com.example.rag.service;

import com.example.rag.embedding.OnnxBgeEmbeddingService;
import com.example.rag.store.VectorStore;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 业务编排：负责把上传文件（文本/CSV）转成向量入库，以及按列返回相似度匹配结果。
 *
 * 设计要点（通用化）：
 * 1. 上传时由调用方声明 `columns`（CSV 哪些列作为结构化元数据 & 拼入嵌入文本）。
 *    代码不识别任何具体业务字段名，所有列名按
 *    上传时的 `columns` 列表动态处理。
 * 2. `columns` 列表之外的 CSV 列被忽略。
 * 3. 检索时由调用方再次声明 `columns`，响应里每个 match 只包含这些列的键值对。
 * 4. 文本上传路径保持不变（`columns=[]` 时走老路）。
 */
@Service
public class RagService {

    private static final Logger log = LoggerFactory.getLogger(RagService.class);

    private final OnnxBgeEmbeddingService embedder;
    private final TextChunker chunker;
    private final VectorStore store;

    public RagService(OnnxBgeEmbeddingService embedder, TextChunker chunker, VectorStore store) {
        this.embedder = embedder;
        this.chunker = chunker;
        this.store = store;
    }

    /**
     * 通用化入库入口。`columns` 列表中每个名字必须与 CSV header 中某列名一致；
     * 对非 CSV 文件该参数被忽略。
     */
    public IngestResult ingest(MultipartFile file, List<String> columns) throws Exception {
        String docId = store.newId();
        String original = file.getOriginalFilename() == null ? "upload.bin" : file.getOriginalFilename();
        List<VectorStore.Entry> entries = new ArrayList<>();
        int produced;
        if (looksLikeCsv(original)) {
            produced = ingestCsv(file, docId, entries, columns);
        } else {
            produced = ingestText(file, docId, entries);
        }
        store.addAll(entries);
        log.info("Ingested file={} kind={} columns={} produced={} totalVectors={}",
                original, looksLikeCsv(original) ? "csv" : "text",
                columns == null ? 0 : columns.size(), produced, store.size());
        return new IngestResult(docId, original, produced, store.size());
    }

    /** 兼容旧调用方（文本路径），等价于 ingest(file, Collections.emptyList())。 */
    public IngestResult ingest(MultipartFile file) throws Exception {
        return ingest(file, Collections.emptyList());
    }

    private int ingestCsv(MultipartFile file, String docId,
                          List<VectorStore.Entry> entries, List<String> columns) throws Exception {
        if (columns == null || columns.isEmpty()) {
            throw new IllegalArgumentException(
                    "CSV ingest requires `columns` to be non-empty (caller must declare which CSV columns to ingest).");
        }
        // 用 BufferedReader 包装：支持 mark/reset，便于剥离 UTF-8 BOM
        try (java.io.BufferedReader br = new java.io.BufferedReader(
                new InputStreamReader(file.getInputStream(), StandardCharsets.UTF_8));
             // 探测并跳过 UTF-8 BOM（如果存在）
             Reader bomStripped = skipUtf8Bom(br);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .setIgnoreEmptyLines(true)
                     .setIgnoreSurroundingSpaces(true)
                     .setQuoteMode(CSVFormat.DEFAULT.getQuoteMode())
                     .build()
                     .parse(bomStripped)) {

            // 校验 columns 列表中的每个名字都在 header 中
            List<String> headerNames = parser.getHeaderNames();
            for (String col : columns) {
                if (!headerNames.contains(col)) {
                    throw new IllegalArgumentException(
                            "Column '" + col + "' not found in CSV header: " + headerNames);
                }
            }

            int produced = 0;
            for (CSVRecord rec : parser) {
                Map<String, String> cols = new LinkedHashMap<>();
                for (String col : columns) {
                    String v = rec.isMapped(col) ? rec.get(col) : null;
                    cols.put(col, v == null ? "" : v);
                }
                // 嵌入文本：按 columns 列表顺序拼接 "key: value"
                StringBuilder sb = new StringBuilder();
                for (String col : columns) {
                    if (sb.length() > 0) sb.append('\n');
                    sb.append(col).append(": ").append(cols.get(col));
                }
                String text = sb.toString();
                float[] vec = embedder.embed(text);
                entries.add(new VectorStore.Entry(store.newId(), docId, text, vec, cols));
                produced++;
            }
            return produced;
        }
    }

    /**
     * 如果 BufferedReader 第一个字符是 U+FEFF（UTF-8 BOM），跳过它；
     * 否则把字符 push 回原位。
     */
    private static Reader skipUtf8Bom(java.io.BufferedReader br) throws java.io.IOException {
        br.mark(1);
        int c = br.read();
        if (c != 0xFEFF) {
            br.reset();
        }
        return br;
    }

    private int ingestText(MultipartFile file, String docId,
                           List<VectorStore.Entry> entries) throws Exception {
        byte[] bytes = file.getBytes();
        String text = new String(bytes, StandardCharsets.UTF_8);
        List<String> chunks = chunker.split(text);
        for (String chunk : chunks) {
            float[] vec = embedder.embed(chunk);
            // 文本路径不携带 columns
            entries.add(new VectorStore.Entry(store.newId(), docId, chunk, vec, Collections.emptyMap()));
        }
        return chunks.size();
    }

    /**
     * 通用化检索。`columns` 必须非空：响应里每个 match 包含这些列的键值对。
     * 内部按 columns 内容拼接成去重 key，相同组合只保留最高分。
     *
     * @return 当 includeScore=true 时返回 `List<Map<String,Object>>`（含 `score`），
     *         否则返回 `List<Map<String,String>>`。
     */
    public List<?> match(String query, int topK, float minScore,
                         boolean includeScore, List<String> columns) throws Exception {
        if (columns == null || columns.isEmpty()) {
            throw new IllegalArgumentException(
                    "Match requires `columns` to be non-empty (caller must declare which columns to return).");
        }
        float[] q = embedder.embed(query);
        List<VectorStore.Hit> raw = store.search(q, Integer.MAX_VALUE);

        // 过滤：必须有 columns 且 columns 包含请求的所有列
        List<VectorStore.Hit> filtered = new ArrayList<>();
        for (VectorStore.Hit h : raw) {
            if (h.columns == null || h.columns.isEmpty()) continue;
            if (h.score < minScore) continue;
            boolean allPresent = true;
            for (String col : columns) {
                if (!h.columns.containsKey(col)) { allPresent = false; break; }
            }
            if (allPresent) filtered.add(h);
        }

        // 按 columns 内容去重
        Map<String, VectorStore.Hit> dedup = new LinkedHashMap<>();
        for (VectorStore.Hit h : filtered) {
            String key = dedupKey(h, columns);
            VectorStore.Hit prev = dedup.get(key);
            if (prev == null || h.score > prev.score) {
                dedup.put(key, h);
            }
        }

        // 排序 + 截断
        List<VectorStore.Hit> sorted = new ArrayList<>(dedup.values());
        sorted.sort((a, b) -> Float.compare(b.score, a.score));
        if (sorted.size() > topK) sorted = sorted.subList(0, topK);

        // 组装响应
        List<Map<String, Object>> out = new ArrayList<>(sorted.size());
        for (VectorStore.Hit h : sorted) {
            Map<String, Object> m = new LinkedHashMap<>();
            for (String col : columns) {
                m.put(col, h.columns.get(col));
            }
            if (includeScore) m.put("score", h.score);
            out.add(m);
        }
        return out;
    }

    private String dedupKey(VectorStore.Hit h, List<String> columns) {
        return columns.stream()
                .map(c -> c + "=" + h.columns.get(c))
                .collect(Collectors.joining(";"));
    }

    public int totalVectors() { return store.size(); }

    /**
     * 旧版相似度检索入口（不要求 columns）。
     * 保留向后兼容：纯文本入库的 Entry 也会被返回（其 columns 为 emptyMap）。
     * CSV 入库的 Entry 也会返回，columns 字段自然带出。
     */
    public List<VectorStore.Hit> search(String query, int topK) throws Exception {
        float[] q = embedder.embed(query);
        return store.search(q, topK);
    }

    public record IngestResult(String docId, String filename, int chunks, int totalVectors) {}

    /**
     * 辅助：把逗号分隔字符串拆为 List，trim + 去重，保留首次出现顺序。
     * 空字符串返回空列表（不含 null）。
     */
    public static List<String> parseColumns(String csv) {
        if (csv == null || csv.isBlank()) return Collections.emptyList();
        List<String> out = new ArrayList<>();
        for (String raw : csv.split(",")) {
            String t = raw.trim();
            if (!t.isEmpty() && !out.contains(t)) out.add(t);
        }
        return out;
    }

    private static boolean looksLikeCsv(String filename) {
        if (filename == null) return false;
        return filename.toLowerCase().endsWith(".csv");
    }
}
