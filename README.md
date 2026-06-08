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

默认 `docs/wiki`（项目根目录下）。支持：
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

同步模式下 CSV 也按文本处理（不分列），整文件经分块后向量化。

## 启动
```bash
mvn spring-boot:run
```
服务端口 `8080`，单文件上传上限 `20MB`。

## 接口
所有接口异常统一由 `GlobalExceptionHandler` 处理，返回 `{"error": "..."}` 的 JSON。

### `POST /api/rag/upload`（multipart/form-data）
文件会先落到 `rag.docs.path`，再向量化入库。

| 字段    | 必传 | 默认        | 说明 |
|---------|------|-------------|------|
| `file`    | 是   | -          | 上传的文本 / CSV / Markdown 文件，不能为空 |
| `columns` | 否   | `column1,column2` | CSV 模式下声明要入库的列（逗号分隔，列名必须出现在 CSV header 中） |
| `name`    | 否   | 原始文件名   | 自定义文档名称（任意非空字符串）。非空时用作 {@code Document.filename}，并校验与现有文档名称不重复（409）；未传时使用上传文件的原始文件名（不做去重） |

### `GET /api/rag/search`
| 参数   | 必传 | 默认 | 范围    | 说明 |
|--------|------|------|---------|------|
| `q`    | 是   | -    | 非空    | 查询文本 |
| `topK` | 否   | 5    | 1..100  | 返回前 K 个 |

### `GET /api/rag/match`
| 参数           | 必传 | 默认              | 范围       | 说明 |
|----------------|------|-------------------|------------|------|
| `q`            | 是   | -                 | 非空       | 查询文本 |
| `columns`      | 是   | `column1,column2`| 非空       | 响应中每个 match 包含的列 |
| `topK`         | 否   | 5                 | 1..100     | 返回前 K 个 |
| `minScore`     | 否   | 0.0               | -1.0..1.0  | 最低分阈值 |
| `includeScore` | 否   | false             | -          | 是否在 match 中包含 `score` 字段 |

### `GET /api/rag/stats`
返回当前向量库的概览：总向量数、文档数、文档列表（docId / name / size / chunks / createdAt）。

### `DELETE /api/rag/documents/{docId}`
按 docId 删除文档：清空对应所有向量 + 删除 `rag.docs.path` 下的原始文件。
classpath 模式下只清空向量。

### `POST /api/rag/sync`
手动触发全量同步：清空当前向量 → 扫描 `rag.docs.path` → 重新入库。
已有同步任务在跑时返回 409。

## 异常响应
| 场景                                  | 状态码 |
|---------------------------------------|--------|
| 入参校验失败 / 缺参 / 类型错误         | 400    |
| 上传文件过大（> 20MB）                 | 413    |
| 路由不存在                             | 404    |
| 文档不存在                             | 404    |
| HTTP 方法不支持                        | 405    |
| 同步任务冲突（已在跑）                  | 409    |
| 文档名称冲突（显式指定 name 已存在）    | 409    |
| 模型未就绪 / 配置错误                  | 503    |
| 其它未捕获异常                         | 500    |

## 设计要点
- **分段**：滑动窗口 + 标点切分，配置 `rag.chunk.size` / `rag.chunk.overlap`
- **池化**：attention-mask aware mean pooling
- **归一化**：L2 normalize，余弦相似度 = 点积
- **存储**：进程内 `VectorStore`，维护 `entries` + `documents` 两种索引
- **同步**：扫描 → 清空 → 重新入库（全量覆盖）
- **互斥**：同步任务用 `AtomicBoolean` 互斥

## 后续可替换
- 替换 `VectorStore` 为 HNSW / Faiss
- 加 rerank 模型
- 加文件类型支持（pdf/docx）
- 同步时支持 CSV 结构化列
