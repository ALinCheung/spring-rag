package com.example.rag.service;

import com.example.rag.chunk.Chunk;
import com.example.rag.chunk.ChunkerRouter;
import com.example.rag.embedding.OnnxBgeEmbeddingService;
import com.example.rag.exception.DocumentConflictException;
import com.example.rag.exception.DocumentNotFoundException;
import com.example.rag.index.DocumentIndex;
import com.example.rag.model.Document;
import com.example.rag.store.DocumentRepository;
import com.example.rag.store.VectorStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 业务编排：
 *  - {@link #ingest(MultipartFile, List, String)}：上传文件落盘 + 向量化入库（主入口）
 *  - {@link #syncAll()}：扫描文档目录，全量覆盖当前内存向量
 *  - {@link #deleteDocument(String)}：按 docId 删向量 + 删原始文件
 *  - {@link #search} / {@link #match}：检索
 */
@Service
public class RagService {

    private static final Logger log = LoggerFactory.getLogger(RagService.class);

    private final OnnxBgeEmbeddingService embedder;
    private final ChunkerRouter chunkerRouter;
    private final VectorStore store;
    private final DocumentRepository docRepo;
    private final DocumentIndex documentIndex;

    public RagService(OnnxBgeEmbeddingService embedder, ChunkerRouter chunkerRouter,
                      VectorStore store, DocumentRepository docRepo, DocumentIndex documentIndex) {
        this.embedder = embedder;
        this.chunkerRouter = chunkerRouter;
        this.store = store;
        this.docRepo = docRepo;
        this.documentIndex = documentIndex;
    }

    // ----- 入库 -----

    /**
     * 通用化入库入口。上传的文件会先落到 rag.docs.path，再向量化入库。
     *
     * @param file    上传的文件（必传，非空）
     * @param columns CSV 模式下声明要入库的列（逗号分隔，列名必须出现在 CSV header 中）。
     *                文本文件忽略此参数。
     * @param name    显式指定的文档名称（可选）。非空时会用作 {@code Document.filename}，并校验
     *                与现有文档名称不重复。{@code null}/空白时回退到上传文件的原始文件名（不做去重）。
     */
    public IngestResult ingest(MultipartFile file, List<String> columns, String name) throws Exception {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("file is required and must not be empty");
        }
        String original = file.getOriginalFilename();
        if (original == null || original.isBlank()) {
            throw new IllegalArgumentException("file must have a valid filename");
        }
        // CSV 上传路径要求 columns 非空,避免误把整张表向量化
        if (original.toLowerCase().endsWith(".csv") && (columns == null || columns.isEmpty())) {
            throw new IllegalArgumentException(
                    "CSV ingest requires `columns` to be non-empty (caller must declare which CSV columns to ingest).");
        }
        String docName = resolveDocName(name, original);
        // 使用 documentIndex 检查文档名称是否重复
        if (name != null && !name.trim().isEmpty() && documentIndex.existsByDocName(docName)) {
            throw new DocumentConflictException(docName);
        }
        DocumentRepository.StoredFile stored = docRepo.save(file);
        byte[] bytes = Files.readAllBytes(stored.path());
        return ingestFromBytes(stored.storedName(), docName, bytes,
                columns == null ? Collections.emptyList() : columns);
    }

    /** 兼容旧调用方：未指定 name，等价于 ingest(file, columns, null)。 */
    public IngestResult ingest(MultipartFile file, List<String> columns) throws Exception {
        return ingest(file, columns, null);
    }

    /** 兼容旧调用方（文本路径），等价于 ingest(file, Collections.emptyList(), null)。 */
    public IngestResult ingest(MultipartFile file) throws Exception {
        return ingest(file, Collections.emptyList(), null);
    }

    /**
     * 决定最终写入 {@code Document.filename} 的名称：
     *  - 显式传入的 name：trim 后非空才使用，防止路径穿越。
     *  - 未传或空白：回退到上传文件的原始文件名。
     */
    private String resolveDocName(String name, String original) {
        if (name == null) return original;
        String trimmed = name.trim();
        if (trimmed.isEmpty()) return original;
        // 防路径穿越：只取 basename 部分
        return Paths.get(trimmed).getFileName().toString();
    }

    /**
     * 把字节内容向量化入库。按文件名后缀路由到对应 Chunker:
     *   .md  -> MarkdownChunker, .csv -> CsvRowChunker, 其他 -> SlidingWindowChunker
     * CSV 在上传路径上要求 columns 非空;同步路径上传 columns 为空,CsvRowChunker 会全列入库。
     *
     * @param storedName   落盘后的文件名（带 UUID 前缀）；同步模式下与 originalName 相同
     * @param originalName 用户可见的原始文件名（用于 Document.filename）
     * @param bytes        文件内容
     * @param columns      CSV 模式下声明要入库的列;同步模式传空列表
     */
    private IngestResult ingestFromBytes(String storedName, String originalName, byte[] bytes,
                                         List<String> columns) throws Exception {
        long startNs = System.nanoTime();
        String docId = store.newId();
        List<Chunk> chunks = chunkerRouter.split(originalName, bytes, columns);
        List<VectorStore.Entry> entries = new ArrayList<>(chunks.size());
        for (Chunk c : chunks) {
            float[] vec = embedder.embed(c.text());
            entries.add(new VectorStore.Entry(store.newId(), docId, c.text(), vec, c.columns()));
        }
        String storagePath = docRepo.isClasspath()
                ? "classpath:" + docRepo.getClasspathRoot() + "/" + storedName
                : docRepo.getFsRoot().resolve(storedName).toString();
        Document doc = new Document(docId, originalName, storagePath,
                bytes.length, entries.size(), Instant.now());
        store.addDocument(doc, entries);
        // 更新文档索引（非同步模式）
        if (!columns.isEmpty()) {
            documentIndex.add(docId, originalName, storedName, bytes.length, entries.size());
        }
        long durationMs = (System.nanoTime() - startNs) / 1_000_000L;
        log.info("Ingested docId={} name='{}' bytes={} chunks={} totalVectors={} duration={}ms",
                docId, originalName, bytes.length, entries.size(), store.size(), durationMs);
        return new IngestResult(docId, originalName, entries.size(), store.size(), durationMs);
    }

    // ----- 同步 -----

    /**
     * 全量同步：清空当前内存向量，再扫描 rag.docs.path 下的所有支持文件并入库。
     * 同步路径下 columns 传空,各 Chunker 按自身默认行为处理(CsvRowChunker 全列入库)。
     * 同步完成后重建文档索引。
     */
    public SyncResult syncAll() {
        long start = System.currentTimeMillis();
        int scanned = 0, ingested = 0, failed = 0;
        List<DocumentRepository.ScannedFile> files;
        try {
            files = docRepo.scan();
        } catch (IOException e) {
            log.error("syncAll: failed to scan documents", e);
            return new SyncResult(0, 0, 0, 0, store.size(),
                    System.currentTimeMillis() - start);
        }
        scanned = files.size();

        store.clear();
        documentIndex.clear();

        for (DocumentRepository.ScannedFile sf : files) {
            // 跳过 index.json 文件本身
            if ("index.json".equals(sf.filename())) {
                log.info("syncAll: skipping index.json");
                continue;
            }
            long fileStart = System.currentTimeMillis();
            try {
                byte[] bytes = docRepo.read(sf);
                ingestFromBytes(sf.filename(), sf.filename(), bytes, Collections.emptyList());
                ingested++;
                long fileMs = System.currentTimeMillis() - fileStart;
                log.info("syncAll: file='{}' ingested in {}ms", sf.filename(), fileMs);
            } catch (Exception e) {
                long fileMs = System.currentTimeMillis() - fileStart;
                log.error("syncAll: failed to ingest file='{}' after {}ms", sf.filename(), fileMs, e);
                failed++;
            }
        }

        // 同步后重建索引（从当前的 documents）
        rebuildIndex();

        long duration = System.currentTimeMillis() - start;
        log.info("Sync done: scanned={} ingested={} failed={} totalVectors={} duration={}ms",
                scanned, ingested, failed, store.size(), duration);
        return new SyncResult(scanned, ingested, 0, failed, store.size(), duration);
    }

    /** 从当前 VectorStore 重建文档索引 */
    private void rebuildIndex() {
        documentIndex.clear();
        List<com.example.rag.index.DocumentIndex.BatchAddEntry> entries = new ArrayList<>();
        for (Document doc : store.listDocuments()) {
            // 从 storagePath 提取文件名
            String fileName = extractFileName(doc.getPath());
            com.example.rag.index.DocumentIndex.BatchAddEntry entry = new com.example.rag.index.DocumentIndex.BatchAddEntry();
            entry.docId = doc.getDocId();
            entry.docName = doc.getFilename();
            entry.fileName = fileName;
            entry.size = doc.getSize();
            entry.chunks = doc.getChunks();
            entry.createdAt = doc.getCreatedAt();
            entries.add(entry);
        }
        documentIndex.addAll(entries);
        log.info("Rebuilt index with {} entries", documentIndex.list().size());
    }

    /** 从存储路径提取文件名 */
    private String extractFileName(String storagePath) {
        if (storagePath == null) return "";
        if (storagePath.startsWith("classpath:")) {
            // classpath:docs/data-raw/xxx.txt
            int lastSlash = storagePath.lastIndexOf('/');
            return lastSlash >= 0 ? storagePath.substring(lastSlash + 1) : storagePath;
        } else {
            // 绝对路径 /path/to/docs/data-raw/xxx.txt
            return Paths.get(storagePath).getFileName().toString();
        }
    }

    // ----- 删除 -----

    /** 按 docId 删除向量与原始文件。classpath 模式下不删文件（仅清空向量条目）。 */
    public DeleteResult deleteDocument(String docId) throws IOException {
        if (docId == null || docId.isBlank()) {
            throw new IllegalArgumentException("docId is required");
        }
        Document doc = store.getDocument(docId);
        if (doc == null) {
            throw new DocumentNotFoundException(docId);
        }
        int removed = store.removeByDocId(docId);
        boolean fileDeleted = false;
        if (!docRepo.isClasspath() && doc.getPath() != null) {
            Path p = Paths.get(doc.getPath()).getFileName();
            if (p != null) {
                fileDeleted = docRepo.delete(p.toString());
            }
        }
        // 从索引中删除
        documentIndex.remove(docId);
        log.info("Deleted docId={} name='{}' removedVectors={} fileDeleted={}",
                docId, doc.getFilename(), removed, fileDeleted);
        return new DeleteResult(docId, doc.getFilename(), removed, fileDeleted, store.size());
    }

    // ----- 检索 -----

    /**
     * 通用化检索。`columns` 必须非空,score 始终返回。
     */
    public List<MatchEntry> match(String query, int topK, float minScore,
                                  List<String> columns, String docId) throws Exception {
        if (columns == null || columns.isEmpty()) {
            throw new IllegalArgumentException(
                    "Match requires `columns` to be non-empty (caller must declare which columns to return).");
        }
        float[] q = embedder.embed(query);
        List<VectorStore.Hit> raw = store.search(q, Integer.MAX_VALUE);

        // 如果指定了 docId，过滤结果
        if (docId != null && !docId.isBlank()) {
            raw = raw.stream()
                    .filter(h -> docId.equals(h.sourceDocId))
                    .collect(java.util.stream.Collectors.toList());
        }

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

        Map<String, VectorStore.Hit> dedup = new LinkedHashMap<>();
        for (VectorStore.Hit h : filtered) {
            String key = dedupKey(h, columns);
            VectorStore.Hit prev = dedup.get(key);
            if (prev == null || h.score > prev.score) {
                dedup.put(key, h);
            }
        }

        List<VectorStore.Hit> sorted = new ArrayList<>(dedup.values());
        sorted.sort((a, b) -> Float.compare(b.score, a.score));
        if (sorted.size() > topK) sorted = sorted.subList(0, topK);

        List<MatchEntry> out = new ArrayList<>(sorted.size());
        for (VectorStore.Hit h : sorted) {
            Map<String, String> cols = new LinkedHashMap<>();
            for (String col : columns) {
                cols.put(col, h.columns.get(col));
            }
            out.add(new MatchEntry(cols, h.score));
        }
        return out;
    }

    /** 兼容旧调用方：无 docId 过滤。 */
    public List<MatchEntry> match(String query, int topK, float minScore,
                                  List<String> columns) throws Exception {
        return match(query, topK, minScore, columns, null);
    }

    private String dedupKey(VectorStore.Hit h, List<String> columns) {
        StringBuilder sb = new StringBuilder();
        for (String c : columns) {
            if (sb.length() > 0) sb.append(';');
            sb.append(c).append('=').append(h.columns.get(c));
        }
        return sb.toString();
    }

    /** 相似度检索:按 minScore 过滤 + 按 docId 过滤 + 截取 topK。不去重、不投影列。 */
    public List<VectorStore.Hit> search(String query, int topK, float minScore, String docId) throws Exception {
        float[] q = embedder.embed(query);
        List<VectorStore.Hit> raw = store.search(q, Integer.MAX_VALUE);

        if (docId != null && !docId.isBlank()) {
            raw = raw.stream()
                    .filter(h -> docId.equals(h.sourceDocId))
                    .collect(java.util.stream.Collectors.toList());
        }
        if (minScore > Float.NEGATIVE_INFINITY) {
            float threshold = minScore;
            raw = raw.stream()
                    .filter(h -> h.score >= threshold)
                    .collect(java.util.stream.Collectors.toList());
        }
        if (raw.size() > topK) {
            raw = raw.subList(0, topK);
        }
        return raw;
    }

    // ----- 概览 -----

    public int totalVectors() { return store.size(); }

    public List<Document> listDocuments() { return store.listDocuments(); }

    public Document getDocument(String docId) { return store.getDocument(docId); }

    // ----- 工具 -----

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

    // ----- 内部值对象 -----

    public record IngestResult(String docId, String name, int chunks, long totalVectors, long durationMs) {}
    public record DeleteResult(String docId, String name, int removedVectors,
                               boolean fileDeleted, long totalVectors) {}
    public record SyncResult(int scanned, int ingested, int removed, int failed,
                             long totalVectors, long durationMs) {}

    /** 匹配单条结果。score 始终返回。 */
    public record MatchEntry(Map<String, String> columns, float score) {}
}
