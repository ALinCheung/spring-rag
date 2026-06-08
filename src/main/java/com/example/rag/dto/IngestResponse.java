package com.example.rag.dto;

import lombok.Getter;

import java.util.List;

/** 上传 / 同步后的入库结果。 */
@Getter
public class IngestResponse {
    private final String docId;
    private final String name;
    private final List<String> columns;
    private final int chunks;
    private final long totalVectors;
    private final long durationMs;

    public IngestResponse(String docId, String name, List<String> columns,
                          int chunks, long totalVectors, long durationMs) {
        this.docId = docId;
        this.name = name;
        this.columns = columns;
        this.chunks = chunks;
        this.totalVectors = totalVectors;
        this.durationMs = durationMs;
    }
}
