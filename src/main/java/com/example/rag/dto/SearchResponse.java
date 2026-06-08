package com.example.rag.dto;

import lombok.Getter;

import java.util.List;

/** 文本路径相似度检索响应。 */
@Getter
public class SearchResponse {
    private final String query;
    private final int topK;
    private final long totalVectors;
    private final List<SearchHitResponse> results;

    public SearchResponse(String query, int topK, long totalVectors,
                          List<SearchHitResponse> results) {
        this.query = query;
        this.topK = topK;
        this.totalVectors = totalVectors;
        this.results = results;
    }
}
