package com.example.rag.dto;

import lombok.Getter;

import java.util.Map;

/** 通用化匹配接口的单条 match。score 始终返回。 */
@Getter
public class MatchEntryResponse {
    private final Map<String, String> columns;
    private final float score;

    public MatchEntryResponse(Map<String, String> columns, float score) {
        this.columns = columns;
        this.score = score;
    }
}
