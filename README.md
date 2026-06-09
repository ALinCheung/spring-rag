# spring-rag

本地 RAG 向量检索 demo：Spring Boot 3.3 + Java 17 + ONNX Runtime + DJL HuggingFace tokenizer。
不依赖任何外部向量数据库，文本向量化、存储、检索全在内存里完成。

## 模型
内置 `bge-small-zh-v1.5`（BGE 中文 512 维 embedding，ONNX + tokenizer），
放在 `src/main/resources/model/bge-small-zh-v1.5/` 下，默认随项目打包。

### 切换为外部模型
修改 `application.yml` 的 `rag.model.path`：
```yaml
rag:
  model:
    path: classpath:model/bge-small-zh-v1.5    # 默认
    # path: /opt/models/bge-small-zh-v1.5       # 外部绝对路径
```
> 目录中必须同时存在 `model.onnx` 与 `tokenizer.json`，否则启动失败并由全局异常处理器返回 503。

## 文档目录
`rag.docs.path` 决定：
- 上传文件的落盘位置
- 启动时 / 定时全量同步的扫描源
- 删除文档时原始文件的删除位置

默认 `docs/data-raw`（项目根目录下）。支持：
- **文件系统路径**（推荐）：`/var/data/wiki` 或相对路径 `docs/wiki`（启动时自动建目录）
- **classpath:** 前缀：只读模式（仅用于同步读取，禁上传 / 删除）

> classpath 模式在 Spring Boot fat jar 中通常只用于演示与单元测试。

## 同步机制
| 触发方式           | 时机                             | 行为 |
|------------------|----------------------------------|------|
| 启动后立即执行     | `ApplicationReadyEvent`          | 受 `rag.docs.sync-on-startup`（默认 true）控制 |
| 定时执行           | `cron = ${rag.docs.sync-cron}`   | 默认 `0 0 * * * *`（每小时整点） |
| HTTP 手动触发      | `POST /api/rag/sync`             | 任意时刻 |

同步语义：**全量覆盖**。每次同步都会清空当前内存向量，再扫描 `rag.docs.path` 重新入库。
同一时刻仅允许一个同步任务在跑；并发触发会被忽略并返回 409。

同步模式与上传走同一套 Chunker 路由：`.md` 按标题切，`.csv` 按行切（同步时 columns 为空 → 全列入库），其他按滑动窗口切。

## 启动
```bash
mvn spring-boot:run
```

> 完整接口文档见 [README-API.md](./README-API.md)。

## 设计要点
- **分段（按文件后缀路由 Chunker）**：`.md` 按标题切（chunk 文本以 `[H1 > H2]` 路径前缀开头）；`.csv` 按行切（首列值作 `row_id`，写入 `columns` 元数据）；其他按滑动窗口 + 标点切分，配置 `rag.chunk.size` / `rag.chunk.overlap`
- **池化**：attention-mask aware mean pooling
- **归一化**：L2 normalize，余弦相似度 = 点积
- **存储**：进程内 `VectorStore`，维护 `entries` + `documents` 两种索引
- **同步**：扫描 → 清空 → 重新入库（全量覆盖）
- **互斥**：同步任务用 `AtomicBoolean` 互斥

## 后续可替换
- 替换 `VectorStore` 为 HNSW / Faiss
- 加 rerank 模型
- 加文件类型支持（pdf/docx）
