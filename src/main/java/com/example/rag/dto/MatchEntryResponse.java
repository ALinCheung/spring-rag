package com.example.rag.dto;

import lombok.Getter;

import java.util.Map;

/** 通用化匹配接口的单条 match。 */
@Getter
public class MatchEntryResponse {
    private final Map<String, String> columns;
    private final Float score;

    public MatchEntryResponse(Map<String, String> columns, Float score) {
        this.columns = columns;
        this.score = score;
    }
}
