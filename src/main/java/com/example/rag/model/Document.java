package com.example.rag.model;

import lombok.AllArgsConstructor;
import lombok.Getter;

import java.time.Instant;

/**
 * 原始文档元数据：与向量库中同一 docId 的若干 Entry 一一对应。
 * 由 {@code DocumentRepository} 落盘到 rag.docs.path，由 {@code VectorStore} 维护内存索引。
 */
@Getter
@AllArgsConstructor
public class Document {

    private final String docId;
    private final String filename;
    private final String path;
    private final long size;
    private final int chunks;
    private final Instant createdAt;

    /** 复制一个新 Document，仅替换 chunks 字段（向量化完成后回填）。 */
    public Document withChunks(int newChunks) {
        return new Document(docId, filename, path, size, newChunks, createdAt);
    }
}
