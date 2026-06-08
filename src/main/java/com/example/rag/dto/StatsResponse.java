package com.example.rag.dto;

import lombok.Getter;

import java.util.List;

/** stats 响应：当前向量库的概览。 */
@Getter
public class StatsResponse {
    private final long totalVectors;
    private final int totalDocuments;
    private final List<DocumentInfoResponse> documents;

    public StatsResponse(long totalVectors, int totalDocuments,
                         List<DocumentInfoResponse> documents) {
        this.totalVectors = totalVectors;
        this.totalDocuments = totalDocuments;
        this.documents = documents;
    }
}
