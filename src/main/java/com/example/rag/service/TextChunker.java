package com.example.rag.service;

import com.example.rag.config.RagProperties;
import org.springframework.stereotype.Component;

import java.util.ArrayList;
import java.util.List;

/**
 * Sliding-window text chunker. We split on sentence-ish punctuation when
 * possible, then pack into chunks of roughly `size` characters with `overlap`
 * characters of carry-over for context continuity.
 *
 * Character-based chunking is a deliberate choice for a v1: it works for
 * Chinese (BGE) and English without pulling in language-specific tokenizers.
 */
@Component
public class TextChunker {

    private final RagProperties props;

    public TextChunker(RagProperties props) {
        this.props = props;
    }

    public List<String> split(String text) {
        if (text == null) return List.of();
        String clean = text.replace("\r\n", "\n").trim();
        if (clean.isEmpty()) return List.of();

        int size = props.getChunk().getSize();
        int overlap = Math.min(props.getChunk().getOverlap(), size / 2);

        List<String> chunks = new ArrayList<>();
        int n = clean.length();
        int start = 0;
        while (start < n) {
            int end = Math.min(start + size, n);
            // Try to break on punctuation within the last 30 chars of the window
            if (end < n) {
                int cut = -1;
                for (int i = end - 1; i > end - 30 && i > start; i--) {
                    char c = clean.charAt(i);
                    if (c == '。' || c == '！' || c == '？' || c == '.' || c == '!' || c == '?' || c == '\n') {
                        cut = i + 1;
                        break;
                    }
                }
                if (cut > start) end = cut;
            }
            String piece = clean.substring(start, end).trim();
            if (!piece.isEmpty()) chunks.add(piece);
            if (end >= n) break;
            start = Math.max(end - overlap, start + 1);
        }
        return chunks;
    }
}
