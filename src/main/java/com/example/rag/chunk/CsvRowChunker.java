package com.example.rag.chunk;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.springframework.stereotype.Component;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 按行切分 CSV。每行一个 chunk,text 格式 "col1: v1\ncol2: v2",
 * columns 至少含 row_id(首列值)。空行跳过;UTF-8 BOM 跳过。
 */
@Component
public class CsvRowChunker implements Chunker {

    @Override
    public List<Chunk> split(byte[] bytes, List<String> columns) throws IOException {
        if (bytes == null || bytes.length == 0) return List.of();
        boolean allColumns = (columns == null || columns.isEmpty());

        try (BufferedReader br = new BufferedReader(
                new InputStreamReader(new ByteArrayInputStream(bytes), StandardCharsets.UTF_8));
             Reader bomStripped = skipUtf8Bom(br);
             CSVParser parser = CSVFormat.DEFAULT.builder()
                     .setHeader()
                     .setSkipHeaderRecord(true)
                     .setIgnoreEmptyLines(true)
                     .setIgnoreSurroundingSpaces(true)
                     .setQuoteMode(CSVFormat.DEFAULT.getQuoteMode())
                     .build()
                     .parse(bomStripped)) {

            List<String> header = parser.getHeaderNames();
            if (!allColumns) {
                for (String col : columns) {
                    if (!header.contains(col)) {
                        throw new IllegalArgumentException(
                                "Column '" + col + "' not found in CSV header: " + header);
                    }
                }
            }
            String firstHeader = header.isEmpty() ? "" : header.get(0);

            List<Chunk> out = new ArrayList<>();
            for (CSVRecord rec : parser) {
                List<String> effective = allColumns ? header : columns;
                Map<String, String> cols = new LinkedHashMap<>();
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < effective.size(); i++) {
                    String col = effective.get(i);
                    String val = rec.isMapped(col) ? rec.get(col) : "";
                    if (val == null) val = "";
                    cols.put(col, val);
                    if (sb.length() > 0) sb.append('\n');
                    sb.append(col).append(": ").append(val);
                }
                String rowId = firstHeader.isEmpty() ? "" : cols.getOrDefault(firstHeader, "");
                Map<String, String> meta = new LinkedHashMap<>();
                meta.put("row_id", rowId == null ? "" : rowId);
                meta.putAll(cols);
                out.add(new Chunk(sb.toString(), meta));
            }
            return out;
        }
    }

    private static Reader skipUtf8Bom(BufferedReader br) throws IOException {
        br.mark(1);
        int c = br.read();
        if (c != 0xFEFF) {
            br.reset();
        }
        return br;
    }
}