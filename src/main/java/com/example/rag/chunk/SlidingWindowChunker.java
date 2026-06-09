package com.example.rag.chunk;

import com.example.rag.config.RagProperties;
import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * 滑动窗口切分器:按 size 字符切,在窗口末尾 30 字符内优先在标点处断。
 * columns 始终为空(MD/CSV 走专用切分器)。
 */
@Component
public class SlidingWindowChunker implements Chunker {

    private final RagProperties props;

    public SlidingWindowChunker(RagProperties props) {
        this.props = props;
    }

    @Override
    public List<Chunk> split(byte[] bytes, List<String> columns) {
        if (bytes == null || bytes.length == 0) return List.of();
        String text = new String(bytes, StandardCharsets.UTF_8);
        return splitText(text);
    }

    private List<Chunk> splitText(String text) {
        String clean = text.replace("\r\n", "\n").trim();
        if (clean.isEmpty()) return List.of();

        int size = props.getChunk().getSize();
        int overlap = Math.min(props.getChunk().getOverlap(), size / 2);

        List<Chunk> chunks = new ArrayList<>();
        int n = clean.length();
        int start = 0;
        while (start < n) {
            int end = Math.min(start + size, n);
            if (end < n) {
                int cut = -1;
                for (int i = end - 1; i > end - 30 && i > start; i--) {
                    char c = clean.charAt(i);
                    if (c == '。' || c == '！' || c == '？' || c == '.'
                            || c == '!' || c == '?' || c == '\n') {
                        cut = i + 1;
                        break;
                    }
                }
                if (cut > start) end = cut;
            }
            String piece = clean.substring(start, end).trim();
            if (!piece.isEmpty()) chunks.add(new Chunk(piece, java.util.Map.of()));
            if (end >= n) break;
            start = Math.max(end - overlap, start + 1);
        }
        return chunks;
    }
}
