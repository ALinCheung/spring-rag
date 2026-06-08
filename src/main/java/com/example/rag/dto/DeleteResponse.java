package com.example.rag.dto;

import lombok.Getter;

/** 文档删除结果。 */
@Getter
public class DeleteResponse {
    private final String docId;
    private final String name;
    private final int removedVectors;
    private final boolean fileDeleted;
    private final long totalVectors;

    public DeleteResponse(String docId, String name, int removedVectors,
                          boolean fileDeleted, long totalVectors) {
        this.docId = docId;
        this.name = name;
        this.removedVectors = removedVectors;
        this.fileDeleted = fileDeleted;
        this.totalVectors = totalVectors;
    }
}
