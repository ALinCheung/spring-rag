package com.example.rag.chunk;

import java.io.IOException;
import java.util.List;

/** 切分器接口。把字节内容解码并切成 Chunk 列表。 */
public interface Chunker {
    /**
     * @param bytes   原始字节(UTF-8 文本或 CSV 等结构化数据)
     * @param columns CSV 切分时用于列过滤;其他格式忽略。null/空 → 全列入库
     * @return 切分结果,不含空 chunk
     */
    List<Chunk> split(byte[] bytes, List<String> columns) throws IOException;
}
