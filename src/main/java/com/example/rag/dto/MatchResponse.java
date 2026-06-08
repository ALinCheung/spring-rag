package com.example.rag.dto;

import lombok.Getter;

import java.util.List;

/** 通用化匹配响应。 */
@Getter
public class MatchResponse {
    private final String query;
    private final List<String> columns;
    private final int topK;
    private final float minScore;
    private final long totalVectors;
    private final List<MatchEntryResponse> matches;

    public MatchResponse(String query, List<String> columns, int topK,
                         float minScore, long totalVectors,
                         List<MatchEntryResponse> matches) {
        this.query = query;
        this.columns = columns;
        this.topK = topK;
        this.minScore = minScore;
        this.totalVectors = totalVectors;
        this.matches = matches;
    }
}
