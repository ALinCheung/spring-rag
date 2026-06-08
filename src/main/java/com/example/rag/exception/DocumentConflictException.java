package com.example.rag.exception;

/**
 * 文档名称冲突：上传时显式指定了 name，但与已存在的文档名称重复。
 * 由 GlobalExceptionHandler 映射为 409 Conflict。
 */
public class DocumentConflictException extends RuntimeException {
    public DocumentConflictException(String name) {
        super("Document name already exists: " + name);
    }
}
