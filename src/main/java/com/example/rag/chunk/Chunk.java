package com.example.rag.chunk;

import java.util.Map;

/** 单个切分片段。text 用于向量化;columns 携带结构化元数据(如 CSV 行 ID)。 */
public record Chunk(String text, Map<String, String> columns) {
    public Chunk {
        columns = columns == null ? Map.of() : Map.copyOf(columns);
    }
}
