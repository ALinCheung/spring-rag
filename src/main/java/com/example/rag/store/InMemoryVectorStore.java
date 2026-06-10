package com.example.rag.store;

import com.example.rag.model.Document;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

/**
 * 进程内向量库：同时维护向量 Entry 与文档元数据 Document。
 * 所有写操作均在写锁内；读操作在读锁内。
 */
public class InMemoryVectorStore implements VectorStore {

    private final List<Entry> entries = new ArrayList<>();
    /** docId → 文档元数据。LinkedHashMap 保证遍历顺序与插入一致。 */
    private final Map<String, Document> documents = new LinkedHashMap<>();
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    @Override
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

    @Override
    public void addAll(List<Entry> items) {
        lock.writeLock().lock();
        try {
            entries.addAll(items);
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
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

    @Override
    public Document getDocument(String docId) {
        lock.readLock().lock();
        try {
            return documents.get(docId);
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
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

    @Override
    public int size() {
        lock.readLock().lock();
        try {
            return entries.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public int documentCount() {
        lock.readLock().lock();
        try {
            return documents.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
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

    @Override
    public void clear() {
        lock.writeLock().lock();
        try {
            entries.clear();
            documents.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public List<String> allDocIds() {
        lock.readLock().lock();
        try {
            return new ArrayList<>(documents.keySet());
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public Set<String> allDocIdSet() {
        lock.readLock().lock();
        try {
            return documents.keySet().stream().collect(Collectors.toUnmodifiableSet());
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
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
