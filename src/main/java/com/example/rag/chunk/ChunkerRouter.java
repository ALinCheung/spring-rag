package com.example.rag.chunk;

import org.springframework.stereotype.Component;

import java.io.IOException;
import java.util.List;

/**
 * 按文件名后缀分派 Chunker:
 *   .md  -> MarkdownChunker(按标题切)
 *   .csv -> CsvRowChunker(按行切;columns 空 -> 全列入库)
 *   其他 -> SlidingWindowChunker(按字符窗口切)
 */
@Component
public class ChunkerRouter {

    private final MarkdownChunker markdownChunker;
    private final CsvRowChunker csvRowChunker;
    private final SlidingWindowChunker slidingWindowChunker;

    public ChunkerRouter(MarkdownChunker markdownChunker,
                         CsvRowChunker csvRowChunker,
                         SlidingWindowChunker slidingWindowChunker) {
        this.markdownChunker = markdownChunker;
        this.csvRowChunker = csvRowChunker;
        this.slidingWindowChunker = slidingWindowChunker;
    }

    public List<Chunk> split(String filename, byte[] bytes, List<String> columns) throws IOException {
        String lower = filename == null ? "" : filename.toLowerCase();
        if (lower.endsWith(".md")) return markdownChunker.split(bytes, columns);
        if (lower.endsWith(".csv")) return csvRowChunker.split(bytes, columns);
        return slidingWindowChunker.split(bytes, columns);
    }
}