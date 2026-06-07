package com.example.rag.store;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Tiny in-memory vector store. Each entry is (id, sourceDocId, text, vector, columns).
 * No external vector DB. Synchronised with a read/write lock.
 *
 * Vectors are assumed to be L2-normalised upstream, so cosine similarity
 * collapses to a plain dot product — fast enough for tens of thousands of
 * vectors on a laptop.
 *
 * `columns` is an immutable map of structured metadata (e.g. CSV column name → value).
 * For plain text ingest, `columns` is an empty map.
 */
public class VectorStore {

    public static class Entry {
        public final String id;
        public final String sourceDocId;
        public final String text;
        public final float[] vector;
        /** 结构化元数据：列名 → 列值。空 Map 表示纯文本入库（无结构化字段）。 */
        public final Map<String, String> columns;

        public Entry(String id, String sourceDocId, String text, float[] vector) {
            this(id, sourceDocId, text, vector, Collections.emptyMap());
        }

        public Entry(String id, String sourceDocId, String text, float[] vector,
                     Map<String, String> columns) {
            this.id = id;
            this.sourceDocId = sourceDocId;
            this.text = text;
            this.vector = vector;
            // 用 unmodifiableMap 防止外部修改，保证 entries 不可变性
            this.columns = columns == null
                    ? Collections.emptyMap()
                    : Collections.unmodifiableMap(columns);
        }
    }

    public static class Hit {
        public final String id;
        public final String sourceDocId;
        public final String text;
        public final float score;
        public final Map<String, String> columns;

        public Hit(String id, String sourceDocId, String text, float score,
                   Map<String, String> columns) {
            this.id = id;
            this.sourceDocId = sourceDocId;
            this.text = text;
            this.score = score;
            this.columns = columns == null
                    ? Collections.emptyMap()
                    : Collections.unmodifiableMap(columns);
        }
    }

    private final List<Entry> entries = new ArrayList<>();
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    public void addAll(List<Entry> items) {
        lock.writeLock().lock();
        try {
            entries.addAll(items);
        } finally {
            lock.writeLock().unlock();
        }
    }

    public int size() {
        lock.readLock().lock();
        try {
            return entries.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    public List<Hit> search(float[] query, int topK) {
        lock.readLock().lock();
        try {
            if (entries.isEmpty() || query == null || query.length == 0) {
                return Collections.emptyList();
            }
            // 命中列表：不过滤（结构化元数据过滤由上层 RagService 处理）
            List<Hit> scored = new ArrayList<>(entries.size());
            for (Entry e : entries) {
                if (e.vector.length != query.length) continue;
                double dot = 0;
                for (int i = 0; i < query.length; i++) {
                    dot += e.vector[i] * query[i];
                }
                scored.add(new Hit(e.id, e.sourceDocId, e.text, (float) dot, e.columns));
            }
            scored.sort((a, b) -> Float.compare(b.score, a.score));
            if (scored.size() > topK) return scored.subList(0, topK);
            return scored;
        } finally {
            lock.readLock().unlock();
        }
    }

    public void clear() {
        lock.writeLock().lock();
        try {
            entries.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }

    public String newId() { return UUID.randomUUID().toString(); }
}
