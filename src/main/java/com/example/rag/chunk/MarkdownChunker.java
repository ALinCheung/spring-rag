package com.example.rag.chunk;

import org.springframework.stereotype.Component;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 按 # / ## / ### 切分 Markdown。遇到代码块围栏时整体作为正文,
 * 围栏内的 # 不视为标题。每个 chunk 文本以 "[祖先标题链]" 开头作为路径前缀。
 */
@Component
public class MarkdownChunker implements Chunker {

    @Override
    public List<Chunk> split(byte[] bytes, List<String> columns) {
        if (bytes == null || bytes.length == 0) return List.of();
        String text = new String(bytes, StandardCharsets.UTF_8);
        return splitText(text);
    }

    private List<Chunk> splitText(String text) {
        String[] lines = text.replace("\r\n", "\n").split("\n", -1);
        List<String> h1Stack = new ArrayList<>();
        List<String> h2Stack = new ArrayList<>();
        List<String> h3Stack = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean inCodeBlock = false;
        List<Chunk> chunks = new ArrayList<>();

        for (String line : lines) {
            String trimmed = line.trim();

            if (trimmed.startsWith("```")) {
                inCodeBlock = !inCodeBlock;
                if (current.length() > 0) current.append('\n');
                current.append(line);
                continue;
            }
            if (inCodeBlock) {
                if (current.length() > 0) current.append('\n');
                current.append(line);
                continue;
            }

            Heading h = parseHeading(trimmed);
            if (h != null) {
                flushChunk(chunks, current, buildPath(h1Stack, h2Stack, h3Stack));
                current.setLength(0);
                applyHeading(h, h1Stack, h2Stack, h3Stack);
                continue;
            }

            if (current.length() > 0 && !trimmed.isEmpty()) current.append('\n');
            if (!trimmed.isEmpty()) current.append(line);
        }
        flushChunk(chunks, current, buildPath(h1Stack, h2Stack, h3Stack));
        return chunks;
    }

    private static void applyHeading(Heading h, List<String> h1, List<String> h2, List<String> h3) {
        switch (h.level) {
            case 1 -> { h1.clear(); h1.add(h.title); h2.clear(); h3.clear(); }
            case 2 -> { h2.clear(); h2.add(h.title); h3.clear(); }
            case 3 -> { h3.clear(); h3.add(h.title); }
            default -> { /* 不处理 H4+ */ }
        }
    }

    private static void flushChunk(List<Chunk> out, StringBuilder body, String path) {
        if (body.length() == 0) return;
        String bodyStr = body.toString().strip();
        if (bodyStr.isEmpty()) return;
        String text = path.isEmpty() ? bodyStr : "[" + path + "]\n" + bodyStr;
        out.add(new Chunk(text, Map.of()));
    }

    private static String buildPath(List<String> h1, List<String> h2, List<String> h3) {
        StringBuilder sb = new StringBuilder();
        if (!h1.isEmpty()) sb.append("# ").append(h1.get(0));
        if (!h2.isEmpty()) { if (sb.length() > 0) sb.append(" > "); sb.append("## ").append(h2.get(0)); }
        if (!h3.isEmpty()) { if (sb.length() > 0) sb.append(" > "); sb.append("### ").append(h3.get(0)); }
        return sb.toString();
    }

    private static Heading parseHeading(String trimmed) {
        if (trimmed.startsWith("# ") && !trimmed.startsWith("## ")) {
            return new Heading(1, trimmed.substring(2).trim());
        }
        if (trimmed.startsWith("## ") && !trimmed.startsWith("### ")) {
            return new Heading(2, trimmed.substring(3).trim());
        }
        if (trimmed.startsWith("### ") && !trimmed.startsWith("#### ")) {
            return new Heading(3, trimmed.substring(4).trim());
        }
        return null;
    }

    private record Heading(int level, String title) {}
}
