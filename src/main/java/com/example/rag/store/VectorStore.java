package com.example.rag.store;

import com.example.rag.model.Document;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * 向量存储接口：定义向量入库、检索、删除等核心操作。
 * 实现类：{@link InMemoryVectorStore}（进程内）、MilvusVectorStore（Milvus 向量数据库）。
 */
public interface VectorStore {

    /**
     * 向量条目：一个文本片段及其向量表示。
     */
    final class Entry {
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

    /**
     * 检索命中结果。
     */
    final class Hit {
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

    // ----- 写操作 -----

    /**
     * 原子地加入一个文档的所有 entry 与元数据。
     */
    void addDocument(Document doc, List<Entry> items);

    /**
     * 兼容旧接口：仅加入 entries 不记录文档元数据。
     */
    void addAll(List<Entry> items);

    /**
     * 按 docId 删除一个文档的全部向量与元数据，返回被删除的向量条数。
     */
    int removeByDocId(String docId);

    /**
     * 清空所有向量与文档元数据。
     */
    void clear();

    // ----- 读操作 -----

    Document getDocument(String docId);

    List<Document> listDocuments();

    int size();

    int documentCount();

    List<Hit> search(float[] query, int topK);

    // ----- 工具方法 -----

    /**
     * 生成唯一 ID。
     */
    default String newId() {
        return UUID.randomUUID().toString();
    }

    /**
     * 收集当前所有 docId。
     */
    List<String> allDocIds();

    /**
     * 收集当前所有 docId 集合的不可变快照。
     */
    default Set<String> allDocIdSet() {
        return allDocIds().stream().collect(Collectors.toUnmodifiableSet());
    }

    /**
     * 判断当前是否已存在指定 filename 的文档（精确匹配）。
     */
    boolean existsByFilename(String filename);
}
