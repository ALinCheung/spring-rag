package com.example.rag.exception;

/**
 * 同步任务冲突：当前已有一次全量同步在跑，重复触发被拒绝。
 * 由 GlobalExceptionHandler 映射为 409 Conflict。
 */
public class SyncConflictException extends RuntimeException {
    public SyncConflictException() {
        super("已有同步任务正在执行，请稍后重试");
    }
}
