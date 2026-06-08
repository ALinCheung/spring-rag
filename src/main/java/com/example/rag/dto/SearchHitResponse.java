package com.example.rag.dto;

import com.example.rag.store.VectorStore;
import lombok.Getter;

import java.util.Map;

/** 文本路径相似度检索的单个 hit。 */
@Getter
public class SearchHitResponse {
    private final String id;
    private final String sourceDocId;
    private final float score;
    private final String text;
    private final Map<String, String> columns;

    public SearchHitResponse(String id, String sourceDocId, float score,
                             String text, Map<String, String> columns) {
        this.id = id;
        this.sourceDocId = sourceDocId;
        this.score = score;
        this.text = text;
        this.columns = columns;
    }

    public static SearchHitResponse from(VectorStore.Hit hit) {
        return new SearchHitResponse(hit.id, hit.sourceDocId, hit.score, hit.text, hit.columns);
    }
}
