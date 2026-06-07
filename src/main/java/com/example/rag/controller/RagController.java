package com.example.rag.controller;

import com.example.rag.service.RagService;
import com.example.rag.store.VectorStore;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/rag")
public class RagController {

    private final RagService ragService;

    public RagController(RagService ragService) {
        this.ragService = ragService;
    }

    /**
     * 上传文件（CSV / 文本）。
     *
     * @param file    上传的文件
     * @param columns CSV 模式下声明要入库的列（逗号分隔，列名必须出现在 CSV header 中）。
     *                文本文件忽略此参数。默认 `column1,column2`。
     */
    @PostMapping(value = "/upload", consumes = "multipart/form-data")
    public ResponseEntity<?> upload(@RequestParam("file") MultipartFile file,
                                    @RequestParam(value = "columns",
                                            defaultValue = "column1,column2") String columns) {
        try {
            List<String> columnList = RagService.parseColumns(columns);
            RagService.IngestResult res = ragService.ingest(file, columnList);
            return ResponseEntity.ok(Map.of(
                    "docId", res.docId(),
                    "filename", res.filename(),
                    "columns", columnList,
                    "chunks", res.chunks(),
                    "totalVectors", res.totalVectors()
            ));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * 旧版相似度检索（文本路径），shape 保持不变，但每个 hit 多了 `columns` 字段（CSV 行才有）。
     */
    @GetMapping("/search")
    public ResponseEntity<?> search(@RequestParam("q") String query,
                                    @RequestParam(value = "topK", defaultValue = "5") int topK) {
        try {
            List<VectorStore.Hit> hits = ragService.search(query, topK);
            List<Map<String, Object>> results = new ArrayList<>(hits.size());
            for (VectorStore.Hit h : hits) {
                Map<String, Object> hitMap = new LinkedHashMap<>();
                hitMap.put("id", h.id);
                hitMap.put("sourceDocId", h.sourceDocId);
                hitMap.put("score", h.score);
                hitMap.put("text", h.text);
                if (h.columns != null && !h.columns.isEmpty()) {
                    hitMap.put("columns", h.columns);
                }
                results.add(hitMap);
            }
            return ResponseEntity.ok(Map.of(
                    "query", query,
                    "topK", topK,
                    "totalVectors", ragService.totalVectors(),
                    "results", results
            ));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }

    @GetMapping("/stats")
    public ResponseEntity<?> stats() {
        return ResponseEntity.ok(Map.of("totalVectors", ragService.totalVectors()));
    }

    /**
     * 通用化匹配端点：返回每个 match 包含请求方声明的 columns 列表对应键值。
     *
     * @param q              查询文本
     * @param columns        逗号分隔的列名列表，响应中每个 match 都包含这些列。
     *                       默认 `column1,column2`。
     * @param topK           返回前 K 个（默认 5）
     * @param minScore       最低分阈值（默认 0.0）
     * @param includeScore   是否在 match 中包含 score 字段（默认 false）
     */
    @GetMapping("/match")
    public ResponseEntity<?> match(@RequestParam("q") String query,
                                   @RequestParam(value = "columns",
                                           defaultValue = "column1,column2") String columns,
                                   @RequestParam(value = "topK", defaultValue = "5") int topK,
                                   @RequestParam(value = "minScore", defaultValue = "0.0") float minScore,
                                   @RequestParam(value = "includeScore", defaultValue = "false") boolean includeScore) {
        try {
            List<String> columnList = RagService.parseColumns(columns);
            if (columnList.isEmpty()) {
                return ResponseEntity.badRequest()
                        .body(Map.of("error", "`columns` must be a non-empty comma-separated list"));
            }
            List<?> matches = ragService.match(query, topK, minScore, includeScore, columnList);
            return ResponseEntity.ok(Map.of(
                    "query", query,
                    "columns", columnList,
                    "topK", topK,
                    "minScore", minScore,
                    "totalVectors", ragService.totalVectors(),
                    "matches", matches
            ));
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(Map.of("error", e.getMessage()));
        } catch (Exception e) {
            return ResponseEntity.internalServerError().body(Map.of("error", e.getMessage()));
        }
    }
}
