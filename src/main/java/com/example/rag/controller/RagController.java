package com.example.rag.controller;

import com.example.rag.dto.DeleteResponse;
import com.example.rag.dto.DocumentInfoResponse;
import com.example.rag.dto.IngestResponse;
import com.example.rag.dto.MatchEntryResponse;
import com.example.rag.dto.MatchResponse;
import com.example.rag.dto.SearchHitResponse;
import com.example.rag.dto.SearchResponse;
import com.example.rag.dto.StatsResponse;
import com.example.rag.dto.SyncResponse;
import com.example.rag.index.DocumentIndex;
import com.example.rag.model.Document;
import com.example.rag.service.RagService;
import com.example.rag.store.VectorStore;
import com.example.rag.sync.DocumentSyncScheduler;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/rag")
public class RagController {

    /** topK 合理上限，防止 OOM / 响应过大。 */
    private static final int MAX_TOP_K = 100;
    /** minScore 合理范围：embedding 已 L2 归一化，余弦相似度 ∈ [-1, 1]。 */
    private static final float MIN_SCORE_LOWER = -1.0f;
    private static final float MIN_SCORE_UPPER = 1.0f;

    private final RagService ragService;
    private final DocumentSyncScheduler syncScheduler;
    private final DocumentIndex documentIndex;

    public RagController(RagService ragService, DocumentSyncScheduler syncScheduler, DocumentIndex documentIndex) {
        this.ragService = ragService;
        this.syncScheduler = syncScheduler;
        this.documentIndex = documentIndex;
    }

    /**
     * 上传文件（CSV / 文本）。文件会先落到 rag.docs.path，再向量化入库。
     */
    @PostMapping(value = "/upload", consumes = "multipart/form-data")
    public IngestResponse upload(@RequestParam("file") MultipartFile file,
                                 @RequestParam(value = "columns",
                                         defaultValue = "column1,column2") String columns,
                                 @RequestParam(value = "name", required = false) String name) throws Exception {
        validateFile(file);
        validateName(name);
        List<String> columnList = RagService.parseColumns(columns);
        RagService.IngestResult res = ragService.ingest(file, columnList, name);
        return new IngestResponse(res.docId(), res.name(), columnList,
                res.chunks(), res.totalVectors(), res.durationMs());
    }

    /**
     * 旧版相似度检索（文本路径），每个 hit 多了 `columns` 字段（CSV 行才有）。
     */
    @GetMapping("/search")
    public SearchResponse search(@RequestParam("q") String query,
                                 @RequestParam(value = "topK", defaultValue = "5") int topK,
                                 @RequestParam(value = "docId", required = false) String docId) throws Exception {
        validateQuery(query);
        validateTopK(topK);
        List<VectorStore.Hit> hits = ragService.search(query, topK, docId);
        List<SearchHitResponse> results = hits.stream()
                .map(SearchHitResponse::from)
                .collect(Collectors.toList());
        return new SearchResponse(query, topK, ragService.totalVectors(), results);
    }

    @GetMapping("/stats")
    public StatsResponse stats() {
        List<Document> docs = ragService.listDocuments();
        List<DocumentInfoResponse> docInfos = docs.stream()
                .map(doc -> {
                    DocumentIndex.IndexEntry entry = documentIndex.get(doc.getDocId());
                    return DocumentInfoResponse.from(doc, entry);
                })
                .collect(Collectors.toList());
        return new StatsResponse(ragService.totalVectors(), docInfos.size(), docInfos);
    }

    /**
     * 通用化匹配端点：返回每个 match 包含请求方声明的 columns 列表对应键值。
     */
    @GetMapping("/match")
    public MatchResponse match(@RequestParam("q") String query,
                               @RequestParam(value = "columns",
                                       defaultValue = "column1,column2") String columns,
                               @RequestParam(value = "topK", defaultValue = "5") int topK,
                               @RequestParam(value = "minScore", defaultValue = "0.0") float minScore,
                               @RequestParam(value = "includeScore", defaultValue = "false") boolean includeScore,
                               @RequestParam(value = "docId", required = false) String docId) throws Exception {
        validateQuery(query);
        validateTopK(topK);
        validateMinScore(minScore);
        List<String> columnList = RagService.parseColumns(columns);
        if (columnList.isEmpty()) {
            throw new IllegalArgumentException("`columns` must be a non-empty comma-separated list");
        }
        List<RagService.MatchEntry> raw = ragService.match(query, topK, minScore, includeScore, columnList, docId);
        List<MatchEntryResponse> matches = new ArrayList<>(raw.size());
        for (RagService.MatchEntry me : raw) {
            matches.add(new MatchEntryResponse(me.columns(), me.score()));
        }
        return new MatchResponse(query, columnList, topK, minScore,
                ragService.totalVectors(), matches);
    }

    /**
     * 按 docId 删除文档：清空对应所有向量，并删除 rag.docs.path 下的原始文件。
     */
    @DeleteMapping("/documents/{docId}")
    public DeleteResponse deleteDocument(@PathVariable("docId") String docId) throws IOException {
        if (docId == null || docId.isBlank()) {
            throw new IllegalArgumentException("`docId` is required");
        }
        RagService.DeleteResult r = ragService.deleteDocument(docId);
        return new DeleteResponse(r.docId(), r.name(), r.removedVectors(),
                r.fileDeleted(), r.totalVectors());
    }

    /**
     * 手动触发全量同步：清空当前内存向量，再扫描 rag.docs.path 重新入库。
     * 若已有同步任务在跑，将抛 409。
     */
    @PostMapping("/sync")
    public SyncResponse sync() {
        RagService.SyncResult r = syncScheduler.triggerSync("manual-http");
        if (r == null) {
            throw new IllegalStateException("已有同步任务正在执行，请稍后重试");
        }
        return new SyncResponse(r.scanned(), r.ingested(), r.removed(), r.failed(),
                r.totalVectors(), r.durationMs());
    }

    // ----- 入参校验（统一抛 IllegalArgumentException，由 GlobalExceptionHandler 映射为 400） -----

    private static void validateFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("`file` is required and must not be empty");
        }
        if (file.getOriginalFilename() == null || file.getOriginalFilename().isBlank()) {
            throw new IllegalArgumentException("`file` must have a valid filename");
        }
        // 禁止上传 index.json 文件（系统内部使用）
        String filename = file.getOriginalFilename();
        if ("index.json".equalsIgnoreCase(filename)) {
            throw new IllegalArgumentException("Uploading 'index.json' is not allowed (system file)");
        }
    }

    private static void validateQuery(String query) {
        if (query == null || query.isBlank()) {
            throw new IllegalArgumentException("`q` is required and must not be blank");
        }
    }

    private static void validateTopK(int topK) {
        if (topK < 1 || topK > MAX_TOP_K) {
            throw new IllegalArgumentException(
                    "`topK` must be between 1 and " + MAX_TOP_K + " (got " + topK + ")");
        }
    }

    private static void validateMinScore(float minScore) {
        if (minScore < MIN_SCORE_LOWER || minScore > MIN_SCORE_UPPER) {
            throw new IllegalArgumentException(
                    "`minScore` must be between " + MIN_SCORE_LOWER + " and " + MIN_SCORE_UPPER
                            + " (got " + minScore + ")");
        }
    }

    /** 校验 name 参数（可选，非空即可）。 */
    private static void validateName(String name) {
        // name 可以是任意非空字符串，不需要扩展名限制
        // 唯一性校验和路径安全检查在 Service 层处理
    }
}
