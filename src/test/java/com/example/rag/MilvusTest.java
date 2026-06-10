package com.example.rag;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.common.DataType;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.collection.request.DropCollectionReq;
import io.milvus.v2.service.collection.request.LoadCollectionReq;
import io.milvus.v2.service.index.request.CreateIndexReq;
import io.milvus.v2.service.utility.request.FlushReq;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.request.data.FloatVec;
import io.milvus.v2.service.vector.response.SearchResp;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Milvus Java SDK 2.x 端到端连通性测试。
 *
 * <p>测试目标：验证本机 localhost:19530 端口上的 Milvus 服务可用，
 * 完成 "建集合 → 入向量 → 向量搜索" 的最小闭环。</p>
 *
 * <p>前置条件：需要本机已启动 Milvus 服务（Standalone / Lite / Docker 任一即可）。</p>
 * <ul>
 *   <li>Standalone Docker: {@code docker run -d --name milvus_standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest}</li>
 * </ul>
 *
 * <p>如果服务未启动，本测试会被 {@code assumeTrue} 优雅跳过（标记为 SKIPPED，
 * 不算失败），不会污染 CI 流水线。</p>
 */
class MilvusTest {

    private static final String HOST = "localhost";
    private static final int PORT = 19530;
    private static final String COLLECTION = "java_db";
    private static final int DIM = 128;

    @Test
    @DisplayName("建集合 → 入向量 → 向量搜索：最小闭环")
    void milvusEndToEndFlow() {
        // 1) 端口连通性预检：避免在 Milvus 未启动时抛一堆无用异常
        boolean reachable;
        try (Socket socket = new Socket()) {
            socket.connect(new InetSocketAddress(HOST, PORT), 1000);
            reachable = true;
        } catch (Exception e) {
            reachable = false;
        }
        assumeTrue(reachable,
                "跳过：localhost:" + PORT + " 上未发现 Milvus 服务。请先启动 Milvus。");

        // 2) 建连接、跑端到端流程；无论成败都确保 dropCollection，try-with-resources 关闭客户端
        MilvusClientV2 client = new MilvusClientV2(ConnectConfig.builder()
                .uri("http://" + HOST + ":" + PORT)
                .build());
        try {
            runEndToEnd(client);
        } finally {
            try {
                client.dropCollection(DropCollectionReq.builder()
                        .collectionName(COLLECTION)
                        .build());
            } catch (Exception ignored) {
                // 清理失败不影响测试结论
            }
            client.close();
        }
    }

    /** 实际执行端到端流程。 */
    private void runEndToEnd(MilvusClientV2 client) {
        // 1) 建集合：1 个 Int64 主键 + 1 个 128 维 FloatVector
        CreateCollectionReq.CollectionSchema schema = client.createSchema();
        schema.addField(AddFieldReq.builder()
                .fieldName("id")
                .dataType(DataType.Int64)
                .isPrimaryKey(true)
                .autoID(false)
                .build());
        schema.addField(AddFieldReq.builder()
                .fieldName("vector")
                .dataType(DataType.FloatVector)
                .dimension(DIM)
                .build());

        client.createCollection(CreateCollectionReq.builder()
                .collectionName(COLLECTION)
                .collectionSchema(schema)
                .build());

        // 2) 显式声明向量字段的索引（HNSW + COSINE），并加载集合到内存
        Map<String, Object> hnswParams = new HashMap<>();
        hnswParams.put("M", 16);
        hnswParams.put("efConstruction", 64);
        client.createIndex(CreateIndexReq.builder()
                .collectionName(COLLECTION)
                .indexParams(Collections.singletonList(
                        IndexParam.builder()
                                .fieldName("vector")
                                .indexType(IndexParam.IndexType.HNSW)
                                .metricType(IndexParam.MetricType.COSINE)
                                .extraParams(hnswParams)
                                .build()))
                .build());
        client.loadCollection(LoadCollectionReq.builder()
                .collectionName(COLLECTION)
                .build());

        // 3) 构造 2 条 128 维向量：第一条全零，第二条以确定性规则递增
        List<JsonObject> rows = new ArrayList<>();
        rows.add(buildRow(1L, makeVector(0L)));
        rows.add(buildRow(2L, makeVector(1L)));

        client.insert(InsertReq.builder()
                .collectionName(COLLECTION)
                .data(rows)
                .build());
        client.flush(FlushReq.builder()
                .collectionNames(Collections.singletonList(COLLECTION))
                .build());

        // 4) 用第一条向量反查自身：top1 应该命中 id=1，相似度近似 1.0
        JsonArray queryVec = rows.get(0).getAsJsonArray("vector");
        SearchResp resp = client.search(SearchReq.builder()
                .collectionName(COLLECTION)
                .annsField("vector")
                .topK(1)
                .data(Collections.singletonList(new FloatVec(jsonArrayToFloats(queryVec))))
                .build());

        List<SearchResp.SearchResult> hits = resp.getSearchResults().get(0);
        assertNotNull(hits, "搜索结果不应为空");
        assertFalse(hits.isEmpty(), "至少应返回 1 条命中");
        SearchResp.SearchResult top = hits.get(0);
        // SDK 2.6.x 的 getEntity() 不包含主键字段，主键应通过 getId() 获取
        assertEquals(1L, ((Number) top.getId()).longValue(),
                "最相似条目应为 id=1（自查询）");
        assertEquals(1.0d, top.getScore().doubleValue(), 1e-3,
                "自查询相似度应接近 1.0");
    }

    /** 构造一条数据行：id + vector（List&lt;Float&gt; 形式，gson 序列化为 JsonArray 后被 Milvus SDK 识别）。 */
    private static JsonObject buildRow(long id, List<Float> vector) {
        JsonObject row = new JsonObject();
        row.addProperty("id", id);

        JsonArray arr = new JsonArray();
        for (Float v : vector) {
            arr.add(v);
        }
        row.add("vector", arr);
        return row;
    }

    /** 生成确定性的 128 维向量：第 i 维 = (seed * 0.001 + i * 0.0001)。 */
    private static List<Float> makeVector(long seed) {
        List<Float> v = new ArrayList<>(DIM);
        for (int i = 0; i < DIM; i++) {
            v.add((float) (seed * 0.001 + i * 0.0001));
        }
        return v;
    }

    private static List<Float> jsonArrayToFloats(JsonArray arr) {
        List<Float> out = new ArrayList<>(arr.size());
        for (int i = 0; i < arr.size(); i++) {
            out.add(arr.get(i).getAsFloat());
        }
        return out;
    }
}
