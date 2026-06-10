package com.example.rag.store;

import com.example.rag.config.RagProperties;
import com.example.rag.embedding.OnnxBgeEmbeddingService;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 向量存储配置：根据 rag.store.type 决定使用内存还是 Milvus。
 * <ul>
 *   <li>{@code memory}（默认）：进程内 {@link InMemoryVectorStore}</li>
 *   <li>{@code milvus}：基于 Milvus 向量数据库的 {@link MilvusVectorStore}</li>
 * </ul>
 */
@Configuration
public class VectorStoreConfig {

    @Bean
    @ConditionalOnProperty(name = "rag.store.type", havingValue = "memory", matchIfMissing = true)
    public VectorStore inMemoryVectorStore() {
        return new InMemoryVectorStore();
    }

    @Bean
    @ConditionalOnProperty(name = "rag.store.type", havingValue = "milvus")
    public VectorStore milvusVectorStore(RagProperties props, OnnxBgeEmbeddingService embeddingService) {
        return new MilvusVectorStore(props, embeddingService);
    }
}
