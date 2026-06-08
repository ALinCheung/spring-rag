package com.example.rag.index;

import com.example.rag.model.Document;
import com.fasterxml.jackson.annotation.JsonFormat;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * 文档索引：维护文档名称（docName）与文件名（fileName）的关系。
 * 持久化到 docs/data-raw/index.json。
 */
@Component
public class DocumentIndex {

    private static final Logger log = LoggerFactory.getLogger(DocumentIndex.class);
    private static final String INDEX_FILE = "index.json";

    private final ObjectMapper objectMapper;
    private final Path indexPath;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    /** 内存缓存：docId → IndexEntry */
    private final Map<String, IndexEntry> entries = new ConcurrentHashMap<>();

    public DocumentIndex(com.example.rag.config.RagProperties props) throws IOException {
        this.objectMapper = new ObjectMapper();
        this.objectMapper.registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule());
        this.objectMapper.enable(SerializationFeature.INDENT_OUTPUT);
        this.objectMapper.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);

        Path docsRoot = props.getDocs().getPath() != null
                ? Path.of(props.getDocs().getPath()).toAbsolutePath().normalize()
                : Path.of("docs/data-raw").toAbsolutePath().normalize();
        this.indexPath = docsRoot.resolve(INDEX_FILE);

        // 确保目录存在
        Files.createDirectories(docsRoot);

        // 加载现有索引
        load();
    }

    /** 加载 index.json 到内存 */
    public void load() {
        lock.writeLock().lock();
        try {
            entries.clear();
            if (!Files.exists(indexPath)) {
                log.info("Index file not found, starting with empty index: {}", indexPath);
                save();
                return;
            }
            byte[] bytes = Files.readAllBytes(indexPath);
            IndexData data = objectMapper.readValue(bytes, IndexData.class);
            if (data.documents != null) {
                for (IndexEntry entry : data.documents) {
                    entries.put(entry.docId, entry);
                }
            }
            log.info("Loaded {} entries from index: {}", entries.size(), indexPath);
        } catch (IOException e) {
            log.error("Failed to load index, starting with empty: {}", indexPath, e);
            entries.clear();
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** 保存内存索引到 index.json */
    public void save() {
        lock.writeLock().lock();
        try {
            IndexData data = new IndexData();
            data.version = "1.0";
            data.documents = new ArrayList<>(entries.values());
            objectMapper.writeValue(indexPath.toFile(), data);
            log.debug("Saved {} entries to index: {}", entries.size(), indexPath);
        } catch (IOException e) {
            log.error("Failed to save index: {}", indexPath, e);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** 添加文档到索引（立即保存） */
    public void add(String docId, String docName, String fileName, long size, int chunks) {
        add(docId, docName, fileName, size, chunks, Instant.now());
        save();
        log.info("Added to index: docId={}, docName={}, fileName={}", docId, docName, fileName);
    }

    /** 添加文档到索引（不保存，用于批量操作） */
    public void add(String docId, String docName, String fileName, long size, int chunks, Instant createdAt) {
        lock.writeLock().lock();
        try {
            IndexEntry entry = new IndexEntry();
            entry.docId = docId;
            entry.docName = docName;
            entry.fileName = fileName;
            entry.size = size;
            entry.chunks = chunks;
            entry.createdAt = createdAt;
            entries.put(docId, entry);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** 批量添加文档到索引（只保存一次） */
    public void addAll(List<BatchAddEntry> entriesToAdd) {
        lock.writeLock().lock();
        try {
            for (BatchAddEntry e : entriesToAdd) {
                IndexEntry entry = new IndexEntry();
                entry.docId = e.docId;
                entry.docName = e.docName;
                entry.fileName = e.fileName;
                entry.size = e.size;
                entry.chunks = e.chunks;
                entry.createdAt = e.createdAt;
                entries.put(e.docId, entry);
            }
            save();
            log.info("Batch added {} entries to index", entriesToAdd.size());
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** 从索引删除文档 */
    public void remove(String docId) {
        lock.writeLock().lock();
        try {
            IndexEntry removed = entries.remove(docId);
            if (removed != null) {
                save();
                log.info("Removed from index: docId={}, docName={}", docId, removed.docName);
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** 清空索引 */
    public void clear() {
        lock.writeLock().lock();
        try {
            entries.clear();
            save();
            log.info("Cleared index");
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** 按 docId 获取索引条目 */
    public IndexEntry get(String docId) {
        lock.readLock().lock();
        try {
            return entries.get(docId);
        } finally {
            lock.readLock().unlock();
        }
    }

    /** 列出所有索引条目 */
    public List<IndexEntry> list() {
        lock.readLock().lock();
        try {
            return new ArrayList<>(entries.values());
        } finally {
            lock.readLock().unlock();
        }
    }

    /** 检查文档名称是否已存在 */
    public boolean existsByDocName(String docName) {
        lock.readLock().lock();
        try {
            for (IndexEntry entry : entries.values()) {
                if (docName.equals(entry.docName)) {
                    return true;
                }
            }
            return false;
        } finally {
            lock.readLock().unlock();
        }
    }

    /** 获取索引文件路径（供外部检查） */
    public Path getIndexPath() {
        return indexPath;
    }

    // ----- 内部数据结构 -----

    /** 批量添加条目 */
    public static class BatchAddEntry {
        public String docId;
        public String docName;
        public String fileName;
        public long size;
        public int chunks;
        public Instant createdAt;
    }

    public static class IndexData {
        public String version;
        public List<IndexEntry> documents;
    }

    public static class IndexEntry {
        public String docId;
        public String docName;
        public String fileName;
        public long size;
        public int chunks;
        @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss'Z'", timezone = "UTC")
        public Instant createdAt;

        /** 从 Document 转换（用于同步场景） */
        public static IndexEntry fromDocument(String fileName, Document doc) {
            IndexEntry entry = new IndexEntry();
            entry.docId = doc.getDocId();
            entry.docName = doc.getFilename();  // Document.filename 就是 docName
            entry.fileName = fileName;
            entry.size = doc.getSize();
            entry.chunks = doc.getChunks();
            entry.createdAt = doc.getCreatedAt();
            return entry;
        }
    }
}
