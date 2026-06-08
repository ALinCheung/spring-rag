package com.example.rag.dto;

import com.example.rag.index.DocumentIndex;
import lombok.Getter;

import java.time.Instant;

/** 文档元数据响应。 */
@Getter
public class DocumentInfoResponse {
    private final String docId;
    private final String docName;
    private final String fileName;
    private final long size;
    private final int chunks;
    private final Instant createdAt;

    public DocumentInfoResponse(String docId, String docName, String fileName,
                                long size, int chunks, Instant createdAt) {
        this.docId = docId;
        this.docName = docName;
        this.fileName = fileName;
        this.size = size;
        this.chunks = chunks;
        this.createdAt = createdAt;
    }

    /** 从 Document 和 IndexEntry 构建 */
    public static DocumentInfoResponse from(com.example.rag.model.Document doc, DocumentIndex.IndexEntry entry) {
        return new DocumentInfoResponse(doc.getDocId(),
                entry != null ? entry.docName : doc.getFilename(),
                entry != null ? entry.fileName : extractFileName(doc.getPath()),
                doc.getSize(), doc.getChunks(), doc.getCreatedAt());
    }

    /** 兼容：仅从 Document 构建（fileName 从 path 提取） */
    public static DocumentInfoResponse from(com.example.rag.model.Document doc) {
        return from(doc, null);
    }

    private static String extractFileName(String path) {
        if (path == null) return "";
        if (path.startsWith("classpath:")) {
            int lastSlash = path.lastIndexOf('/');
            return lastSlash >= 0 ? path.substring(lastSlash + 1) : path;
        }
        return java.nio.file.Paths.get(path).getFileName().toString();
    }
}
