package com.example.rag.config;

import lombok.Data;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Configuration;

@Data
@Configuration
@ConfigurationProperties(prefix = "rag")
public class RagProperties {

    private Model model = new Model();
    private Chunk chunk = new Chunk();
    private Docs docs = new Docs();

    @Data
    public static class Model {
        private String path;
    }

    @Data
    public static class Chunk {
        private int size = 200;
        private int overlap = 30;
    }

    /** 原始文档存储目录 + 定时同步配置。 */
    @Data
    public static class Docs {
        /** 文档目录。支持 classpath: 前缀（仅 IDE 有效；生产建议用文件系统路径）。默认 docs/data-raw。 */
        private String path = "docs/data-raw";
        /** 定时同步 cron 表达式，默认每小时整点。设为 `disable` 或空字符串可关闭定时同步。 */
        private String syncCron = "0 0 * * * *";
        /** 启动时是否立即执行一次全量同步。 */
        private boolean syncOnStartup = true;
        /** 单文档最大字节数（防止内存爆掉），默认 20MB。 */
        private long maxFileSize = 20L * 1024 * 1024;
    }
}
