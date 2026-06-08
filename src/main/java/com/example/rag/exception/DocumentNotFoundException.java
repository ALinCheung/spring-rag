package com.example.rag.exception;

/** 文档不存在时抛出，由 GlobalExceptionHandler 映射为 404。 */
public class DocumentNotFoundException extends RuntimeException {
    public DocumentNotFoundException(String docId) {
        super("Document not found: " + docId);
    }
}
