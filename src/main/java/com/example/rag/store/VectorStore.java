package com.example.rag.store;

import com.example.rag.model.Document;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

/**
 * 进程内向量库：同时维护向量 Entry 与文档元数据 Document。
 * 所有写操作均在写锁内；读操作在读锁内。
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
    /** docId → 文档元数据。LinkedHashMap 保证遍历顺序与插入一致。 */
    private final Map<String, Document> documents = new LinkedHashMap<>();
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    /**
     * 原子地加入一个文档的所有 entry 与元数据。
     * 文档元数据记录的是本次 add 的 chunk 数量。
     */
    public void addDocument(Document doc, List<Entry> items) {
        if (doc == null) throw new IllegalArgumentException("doc must not be null");
        lock.writeLock().lock();
        try {
            entries.addAll(items);
            documents.put(doc.getDocId(), doc);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * 兼容旧接口：仅加入 entries 不记录文档元数据。
     * 主要用于旧单元测试；业务侧请改用 addDocument。
     */
    public void addAll(List<Entry> items) {
        lock.writeLock().lock();
        try {
            entries.addAll(items);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** 按 docId 删除一个文档的全部向量与元数据，返回被删除的向量条数。 */
    public int removeByDocId(String docId) {
        lock.writeLock().lock();
        try {
            Document removed = documents.remove(docId);
            if (removed == null) return 0;
            int before = entries.size();
            entries.removeIf(e -> docId.equals(e.sourceDocId));
            return before - entries.size();
        } finally {
            lock.writeLock().unlock();
        }
    }

    public Document getDocument(String docId) {
        lock.readLock().lock();
        try {
            return documents.get(docId);
        } finally {
            lock.readLock().unlock();
        }
    }

    /** 列出全部文档元数据，按 createdAt 升序。 */
    public List<Document> listDocuments() {
        lock.readLock().lock();
        try {
            List<Document> out = new ArrayList<>(documents.values());
            out.sort(Comparator.comparing(Document::getCreatedAt));
            return out;
        } finally {
            lock.readLock().unlock();
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

    public int documentCount() {
        lock.readLock().lock();
        try {
            return documents.size();
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
            documents.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }

    public String newId() { return UUID.randomUUID().toString(); }

    /** 收集当前所有 docId。供 sync 等场景做"已存在"判断。 */
    public List<String> allDocIds() {
        lock.readLock().lock();
        try {
            return new ArrayList<>(documents.keySet());
        } finally {
            lock.readLock().unlock();
        }
    }

    /** 收集当前所有 docId 集合的不可变快照。 */
    public java.util.Set<String> allDocIdSet() {
        lock.readLock().lock();
        try {
            return documents.keySet().stream().collect(Collectors.toUnmodifiableSet());
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * 判断当前是否已存在指定 filename 的文档（精确匹配）。
     * 用于上传时显式指定文档名称的唯一性校验。
     */
    public boolean existsByFilename(String filename) {
        if (filename == null) return false;
        lock.readLock().lock();
        try {
            for (Document d : documents.values()) {
                if (filename.equals(d.getFilename())) return true;
            }
            return false;
        } finally {
            lock.readLock().unlock();
        }
    }
}
