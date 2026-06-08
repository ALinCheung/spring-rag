package com.example.rag.dto;

import lombok.Getter;

/** 同步任务结果。 */
@Getter
public class SyncResponse {
    private final int scanned;
    private final int ingested;
    private final int removed;
    private final int failed;
    private final long totalVectors;
    private final long durationMs;

    public SyncResponse(int scanned, int ingested, int removed,
                        int failed, long totalVectors, long durationMs) {
        this.scanned = scanned;
        this.ingested = ingested;
        this.removed = removed;
        this.failed = failed;
        this.totalVectors = totalVectors;
        this.durationMs = durationMs;
    }
}
