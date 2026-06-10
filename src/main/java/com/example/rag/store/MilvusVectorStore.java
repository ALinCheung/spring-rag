package com.example.rag.store;

import com.example.rag.config.RagProperties;
import com.example.rag.embedding.OnnxBgeEmbeddingService;
import com.example.rag.model.Document;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.collection.request.GetCollectionStatsReq;
import io.milvus.v2.service.collection.request.HasCollectionReq;
import io.milvus.v2.service.collection.request.LoadCollectionReq;
import io.milvus.v2.service.index.request.CreateIndexReq;
import io.milvus.v2.service.utility.request.FlushReq;
import io.milvus.v2.service.vector.request.DeleteReq;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.request.QueryReq;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.response.DeleteResp;
import io.milvus.v2.service.vector.response.QueryResp;
import io.milvus.v2.service.vector.response.SearchResp;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.extern.slf4j.Slf4j;

import java.lang.reflect.Type;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 基于 Milvus 向量数据库的 VectorStore 实现。
 * 使用 milvus-sdk-java 2.x V2 API，维护 vectors 与 documents 两个集合。
 */
@Slf4j
public class MilvusVectorStore implements VectorStore {

    private static final String FIELD_ID = "id";
    private static final String FIELD_SOURCE_DOC_ID = "source_doc_id";
    private static final String FIELD_TEXT = "text";
    private static final String FIELD_VECTOR = "vector";
    private static final String FIELD_COLUMNS_JSON = "columns_json";

    private static final String FIELD_DOC_ID = "doc_id";
    private static final String FIELD_FILENAME = "filename";
    private static final String FIELD_PATH = "path";
    private static final String FIELD_SIZE = "size";
    private static final String FIELD_CHUNKS = "chunks";
    private static final String FIELD_CREATED_AT = "created_at";

    private static final long QUERY_ALL_LIMIT = 16_384L;
    private static final Type MAP_TYPE = new TypeToken<Map<String, String>>() {
    }.getType();

    private final RagProperties props;
    private final OnnxBgeEmbeddingService embeddingService;
    private final Gson gson = new Gson();

    private MilvusClientV2 client;
    private String vectorsCollection;
    private String documentsCollection;
    private int dimension;

    public MilvusVectorStore(RagProperties props, OnnxBgeEmbeddingService embeddingService) {
        this.props = props;
        this.embeddingService = embeddingService;
    }

    @PostConstruct
    public void init() {
        RagProperties.Milvus milvus = props.getStore().getMilvus();
        this.vectorsCollection = milvus.getCollection() + "_vectors";
        this.documentsCollection = milvus.getCollection() + "_documents";
        this.dimension = milvus.getDimension();

        String uri = "http://" + milvus.getHost() + ":" + milvus.getPort();
        try {
            client = new MilvusClientV2(ConnectConfig.builder().uri(uri).build());
        } catch (Exception e) {
            log.error("无法连接到 Milvus: {}", uri, e);
            throw new IllegalStateException("无法连接到 Milvus: " + uri + "，请确认服务已启动且配置正确", e);
        }

        try {
            initVectorsCollection();
            initDocumentsCollection();
        } catch (Exception e) {
            log.error("初始化 Milvus 集合失败", e);
            throw new IllegalStateException("初始化 Milvus 集合失败: " + e.getMessage(), e);
        }
    }

    @PreDestroy
    public void close() {
        if (client != null) {
            try {
                client.close();
            } catch (Exception e) {
                log.error("关闭 Milvus 客户端失败", e);
            } finally {
                client = null;
            }
        }
    }

    private void initVectorsCollection() {
        if (!hasCollection(vectorsCollection)) {
            CreateCollectionReq.CollectionSchema schema = client.createSchema();
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_ID)
                    .dataType(DataType.VarChar)
                    .maxLength(64)
                    .isPrimaryKey(true)
                    .autoID(false)
                    .build());
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_SOURCE_DOC_ID)
                    .dataType(DataType.VarChar)
                    .maxLength(64)
                    .build());
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_TEXT)
                    .dataType(DataType.VarChar)
                    .maxLength(65535)
                    .build());
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_VECTOR)
                    .dataType(DataType.FloatVector)
                    .dimension(dimension)
                    .build());
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_COLUMNS_JSON)
                    .dataType(DataType.VarChar)
                    .maxLength(65535)
                    .build());

            client.createCollection(CreateCollectionReq.builder()
                    .collectionName(vectorsCollection)
                    .collectionSchema(schema)
                    .build());

            Map<String, Object> hnswParams = new HashMap<>();
            hnswParams.put("M", 16);
            hnswParams.put("efConstruction", 64);
            client.createIndex(CreateIndexReq.builder()
                    .collectionName(vectorsCollection)
                    .indexParams(Collections.singletonList(
                            IndexParam.builder()
                                    .fieldName(FIELD_VECTOR)
                                    .indexType(IndexParam.IndexType.HNSW)
                                    .metricType(IndexParam.MetricType.COSINE)
                                    .extraParams(hnswParams)
                                    .build()))
                    .build());
        }
        client.loadCollection(LoadCollectionReq.builder().collectionName(vectorsCollection).build());
    }

    private void initDocumentsCollection() {
        if (!hasCollection(documentsCollection)) {
            CreateCollectionReq.CollectionSchema schema = client.createSchema();
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_DOC_ID)
                    .dataType(DataType.VarChar)
                    .maxLength(64)
                    .isPrimaryKey(true)
                    .autoID(false)
                    .build());
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_FILENAME)
                    .dataType(DataType.VarChar)
                    .maxLength(512)
                    .build());
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_PATH)
                    .dataType(DataType.VarChar)
                    .maxLength(1024)
                    .build());
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_SIZE)
                    .dataType(DataType.Int64)
                    .build());
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_CHUNKS)
                    .dataType(DataType.Int32)
                    .build());
            schema.addField(AddFieldReq.builder()
                    .fieldName(FIELD_CREATED_AT)
                    .dataType(DataType.Int64)
                    .build());
            // Milvus 要求每个 collection 必须包含向量字段。
            // documents collection 不做向量检索，用 2 维 dummy 向量满足约束。
            schema.addField(AddFieldReq.builder()
                    .fieldName("dummy_vector")
                    .dataType(DataType.FloatVector)
                    .dimension(2)
                    .build());

            client.createCollection(CreateCollectionReq.builder()
                    .collectionName(documentsCollection)
                    .collectionSchema(schema)
                    .build());

            // 为 dummy_vector 创建索引以满足 Milvus 要求
            client.createIndex(CreateIndexReq.builder()
                    .collectionName(documentsCollection)
                    .indexParams(Collections.singletonList(
                            IndexParam.builder()
                                    .fieldName("dummy_vector")
                                    .indexType(IndexParam.IndexType.HNSW)
                                    .metricType(IndexParam.MetricType.COSINE)
                                    .extraParams(Map.of("M", 4, "efConstruction", 8))
                                    .build()))
                    .build());
        }
        client.loadCollection(LoadCollectionReq.builder().collectionName(documentsCollection).build());
    }

    private boolean hasCollection(String name) {
        return Boolean.TRUE.equals(client.hasCollection(HasCollectionReq.builder()
                .collectionName(name)
                .build()));
    }

    @Override
    public void addDocument(Document doc, List<Entry> items) {
        if (doc == null) {
            throw new IllegalArgumentException("doc must not be null");
        }
        try {
            if (items != null && !items.isEmpty()) {
                List<JsonObject> rows = new ArrayList<>(items.size());
                for (Entry e : items) {
                    rows.add(toVectorRow(e));
                }
                client.insert(InsertReq.builder()
                        .collectionName(vectorsCollection)
                        .data(rows)
                        .build());
            }

            JsonObject docRow = new JsonObject();
            docRow.addProperty(FIELD_DOC_ID, doc.getDocId());
            docRow.addProperty(FIELD_FILENAME, doc.getFilename());
            docRow.addProperty(FIELD_PATH, doc.getPath() == null ? "" : doc.getPath());
            docRow.addProperty(FIELD_SIZE, doc.getSize());
            docRow.addProperty(FIELD_CHUNKS, doc.getChunks());
            docRow.addProperty(FIELD_CREATED_AT, doc.getCreatedAt().toEpochMilli());
            // dummy_vector: Milvus 要求每个 collection 必须有向量字段
            docRow.add("dummy_vector", gson.toJsonTree(new float[]{0f, 0f}));
            client.insert(InsertReq.builder()
                    .collectionName(documentsCollection)
                    .data(Collections.singletonList(docRow))
                    .build());

            flushQuietly(List.of(vectorsCollection, documentsCollection));
        } catch (Exception e) {
            log.error("addDocument 失败: docId={}", doc.getDocId(), e);
            throw new IllegalStateException("addDocument 失败: " + e.getMessage(), e);
        }
    }

    @Override
    public void addAll(List<Entry> items) {
        if (items == null || items.isEmpty()) {
            return;
        }
        try {
            List<JsonObject> rows = new ArrayList<>(items.size());
            for (Entry e : items) {
                rows.add(toVectorRow(e));
            }
            client.insert(InsertReq.builder()
                    .collectionName(vectorsCollection)
                    .data(rows)
                    .build());
            flushQuietly(Collections.singletonList(vectorsCollection));
        } catch (Exception e) {
            log.error("addAll 失败", e);
            throw new IllegalStateException("addAll 失败: " + e.getMessage(), e);
        }
    }

    private JsonObject toVectorRow(Entry e) {
        JsonObject row = new JsonObject();
        row.addProperty(FIELD_ID, e.id);
        row.addProperty(FIELD_SOURCE_DOC_ID, e.sourceDocId);
        row.addProperty(FIELD_TEXT, e.text);
        row.add(FIELD_VECTOR, gson.toJsonTree(e.vector));
        row.addProperty(FIELD_COLUMNS_JSON, gson.toJson(e.columns));
        return row;
    }

    @Override
    public int removeByDocId(String docId) {
        if (docId == null) {
            return 0;
        }
        try {
            client.query(QueryReq.builder()
                    .collectionName(vectorsCollection)
                    .filter(FIELD_SOURCE_DOC_ID + " == '" + escape(docId) + "'")
                    .outputFields(Collections.singletonList(FIELD_ID))
                    .limit(QUERY_ALL_LIMIT)
                    .build());

            DeleteResp vectorDelete = client.delete(DeleteReq.builder()
                    .collectionName(vectorsCollection)
                    .filter(FIELD_SOURCE_DOC_ID + " == '" + escape(docId) + "'")
                    .build());

            client.delete(DeleteReq.builder()
                    .collectionName(documentsCollection)
                    .filter(FIELD_DOC_ID + " == '" + escape(docId) + "'")
                    .build());

            return (int) vectorDelete.getDeleteCnt();
        } catch (Exception e) {
            log.error("removeByDocId 失败: docId={}", docId, e);
            return 0;
        }
    }

    @Override
    public Document getDocument(String docId) {
        if (docId == null) {
            return null;
        }
        try {
            QueryResp resp = client.query(QueryReq.builder()
                    .collectionName(documentsCollection)
                    .filter(FIELD_DOC_ID + " == '" + escape(docId) + "'")
                    .outputFields(List.of(FIELD_DOC_ID, FIELD_FILENAME, FIELD_PATH, FIELD_SIZE, FIELD_CHUNKS, FIELD_CREATED_AT))
                    .limit(1)
                    .build());
            List<QueryResp.QueryResult> results = resp.getQueryResults();
            if (results == null || results.isEmpty()) {
                return null;
            }
            return toDocument(results.get(0).getEntity());
        } catch (Exception e) {
            log.error("getDocument 失败: docId={}", docId, e);
            return null;
        }
    }

    @Override
    public List<Document> listDocuments() {
        try {
            QueryResp resp = client.query(QueryReq.builder()
                    .collectionName(documentsCollection)
                    .filter(FIELD_DOC_ID + " != ''")
                    .outputFields(List.of(FIELD_DOC_ID, FIELD_FILENAME, FIELD_PATH, FIELD_SIZE, FIELD_CHUNKS, FIELD_CREATED_AT))
                    .limit(QUERY_ALL_LIMIT)
                    .build());
            List<Document> docs = new ArrayList<>();
            List<QueryResp.QueryResult> results = resp.getQueryResults();
            if (results != null) {
                for (QueryResp.QueryResult r : results) {
                    Document d = toDocument(r.getEntity());
                    if (d != null) {
                        docs.add(d);
                    }
                }
            }
            docs.sort(Comparator.comparing(Document::getCreatedAt));
            return docs;
        } catch (Exception e) {
            log.error("listDocuments 失败", e);
            return Collections.emptyList();
        }
    }

    @Override
    public int size() {
        try {
            return client.getCollectionStats(GetCollectionStatsReq.builder()
                            .collectionName(vectorsCollection)
                            .build())
                    .getNumOfEntities()
                    .intValue();
        } catch (Exception e) {
            log.error("获取 vectors 集合行数失败", e);
            return 0;
        }
    }

    @Override
    public int documentCount() {
        try {
            return client.getCollectionStats(GetCollectionStatsReq.builder()
                            .collectionName(documentsCollection)
                            .build())
                    .getNumOfEntities()
                    .intValue();
        } catch (Exception e) {
            log.error("获取 documents 集合行数失败", e);
            return 0;
        }
    }

    /** Milvus topK 上限，超过此值会被钳制 */
    private static final int MILVUS_MAX_TOP_K = 16_384;

    @Override
    public List<Hit> search(float[] query, int topK) {
        if (query == null || query.length == 0 || topK <= 0) {
            return Collections.emptyList();
        }
        // RagService 会传入 Integer.MAX_VALUE 意图取全部结果，Milvus 最大只允许 16384
        int effectiveTopK = Math.min(topK, MILVUS_MAX_TOP_K);
        try {
            SearchReq searchReq = SearchReq.builder()
                    .collectionName(vectorsCollection)
                    .annsField(FIELD_VECTOR)
                    .topK(effectiveTopK)
                    .metricType(IndexParam.MetricType.COSINE)
                    .data(Collections.singletonList(new FloatVec(query)))
                    .outputFields(List.of(FIELD_ID, FIELD_SOURCE_DOC_ID, FIELD_TEXT, FIELD_COLUMNS_JSON))
                    .build();

            SearchResp resp = client.search(searchReq);

            List<List<SearchResp.SearchResult>> groups = resp.getSearchResults();
            if (groups == null || groups.isEmpty()) {
                return Collections.emptyList();
            }

            List<Hit> hits = new ArrayList<>(groups.get(0).size());
            for (SearchResp.SearchResult r : groups.get(0)) {
                Map<String, Object> entity = r.getEntity();
                String columnsJson = entity.get(FIELD_COLUMNS_JSON) == null
                        ? "{}"
                        : entity.get(FIELD_COLUMNS_JSON).toString();
                Map<String, String> columns = gson.fromJson(columnsJson, MAP_TYPE);
                if (columns == null) {
                    columns = Collections.emptyMap();
                }
                String id = r.getId() == null ? "" : r.getId().toString();
                String sourceDocId = entity.get(FIELD_SOURCE_DOC_ID) == null
                        ? ""
                        : entity.get(FIELD_SOURCE_DOC_ID).toString();
                String text = entity.get(FIELD_TEXT) == null ? "" : entity.get(FIELD_TEXT).toString();
                float score = r.getScore() == null ? 0f : r.getScore();
                hits.add(new Hit(id, sourceDocId, text, score, columns));
            }
            hits.sort((a, b) -> Float.compare(b.score, a.score));
            return hits;
        } catch (Exception e) {
            log.error("向量搜索失败", e);
            return Collections.emptyList();
        }
    }

    @Override
    public void clear() {
        try {
            client.delete(DeleteReq.builder()
                    .collectionName(vectorsCollection)
                    .filter(FIELD_ID + " != ''")
                    .build());
        } catch (Exception e) {
            log.error("清空 vectors 集合失败", e);
        }
        try {
            client.delete(DeleteReq.builder()
                    .collectionName(documentsCollection)
                    .filter(FIELD_DOC_ID + " != ''")
                    .build());
        } catch (Exception e) {
            log.error("清空 documents 集合失败", e);
        }
    }

    @Override
    public List<String> allDocIds() {
        try {
            QueryResp resp = client.query(QueryReq.builder()
                    .collectionName(documentsCollection)
                    .filter(FIELD_DOC_ID + " != ''")
                    .outputFields(Collections.singletonList(FIELD_DOC_ID))
                    .limit(QUERY_ALL_LIMIT)
                    .build());
            List<String> ids = new ArrayList<>();
            List<QueryResp.QueryResult> results = resp.getQueryResults();
            if (results != null) {
                for (QueryResp.QueryResult r : results) {
                    Object v = r.getEntity().get(FIELD_DOC_ID);
                    if (v != null) {
                        ids.add(v.toString());
                    }
                }
            }
            return ids;
        } catch (Exception e) {
            log.error("allDocIds 失败", e);
            return Collections.emptyList();
        }
    }

    @Override
    public boolean existsByFilename(String filename) {
        if (filename == null) {
            return false;
        }
        try {
            QueryResp resp = client.query(QueryReq.builder()
                    .collectionName(documentsCollection)
                    .filter(FIELD_FILENAME + " == '" + escape(filename) + "'")
                    .outputFields(Collections.singletonList(FIELD_DOC_ID))
                    .limit(1)
                    .build());
            List<QueryResp.QueryResult> results = resp.getQueryResults();
            return results != null && !results.isEmpty();
        } catch (Exception e) {
            log.error("existsByFilename 失败: filename={}", filename, e);
            return false;
        }
    }

    private Document toDocument(Map<String, Object> entity) {
        if (entity == null) {
            return null;
        }
        String docId = entity.get(FIELD_DOC_ID) == null ? "" : entity.get(FIELD_DOC_ID).toString();
        String filename = entity.get(FIELD_FILENAME) == null ? "" : entity.get(FIELD_FILENAME).toString();
        String path = entity.get(FIELD_PATH) == null ? "" : entity.get(FIELD_PATH).toString();
        long size = 0L;
        Object sizeObj = entity.get(FIELD_SIZE);
        if (sizeObj instanceof Number) {
            size = ((Number) sizeObj).longValue();
        }
        int chunks = 0;
        Object chunksObj = entity.get(FIELD_CHUNKS);
        if (chunksObj instanceof Number) {
            chunks = ((Number) chunksObj).intValue();
        }
        long created = 0L;
        Object createdObj = entity.get(FIELD_CREATED_AT);
        if (createdObj instanceof Number) {
            created = ((Number) createdObj).longValue();
        }
        return new Document(docId, filename, path, size, chunks, Instant.ofEpochMilli(created));
    }

    /**
     * 安静地 flush，不因 Milvus RateLimiter 中断调用流程。
     * Milvus Standalone 有请求频率限制，高频 flush 会被限流；
     * 即使 flush 失败，Milvus 也会在后台自动 flush（默认约 1 秒），数据最终可达。
     */
    private void flushQuietly(List<String> collectionNames) {
        try {
            client.flush(FlushReq.builder()
                    .collectionNames(collectionNames)
                    .build());
        } catch (Exception e) {
            log.warn("flush 失败（Milvus 会自动重试）: {}", e.getMessage());
        }
    }

    private String escape(String value) {
        return value.replace("\\", "\\\\").replace("'", "\\'");
    }
}
