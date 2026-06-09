# 接口文档

本文档描述 spring-rag 服务的全部 HTTP 接口与异常响应规范。
服务端口 `8080`，单文件上传上限 `20MB`。

> 全部接口异常统一由 `GlobalExceptionHandler` 处理，返回 `{"error": "..."}` 的 JSON。

## 接口

### `POST /api/rag/upload`（multipart/form-data）
文件会先落到 `rag.docs.path`，再向量化入库。

| 字段    | 必传 | 默认        | 说明 |
|---------|------|-------------|------|
| `file`    | 是   | -          | 上传的文本 / CSV / Markdown 文件，不能为空 |
| `columns` | 否   | `column1,column2` | CSV 模式下声明要入库的列（逗号分隔，列名必须出现在 CSV header 中） |
| `name`    | 否   | 原始文件名   | 自定义文档名称（任意非空字符串）。非空时用作 {@code Document.filename}，并校验与现有文档名称不重复（409）；未传时使用上传文件的原始文件名（不做去重） |

### `GET /api/rag/search`
通用化检索端点。根据 `columns` 是否传值走两种模式:

**模式一:全文检索**(不传 `columns` 或 `columns=` 空)
- 响应:`{query, topK, totalVectors, results: [{id, sourceDocId, score, text, columns}]}`
- 每个 hit 含 `text` 与所有 metadata 列;`score` 始终返回;不去重、不列投影
- `minScore` 默认 `0.0`(不过滤)

**模式二:列化匹配**(`columns=id,name,...` 非空)
- 响应:`{query, columns, topK, minScore, totalVectors, matches: [{columns, score}]}`
- 每个 match **不含** `text`,只含请求的列;按 `columns` 组合 key dedup;`score` 始终返回
- `minScore` 默认 `0.0`

| 参数           | 必传 | 默认 | 范围       | 说明 |
|----------------|------|------|------------|------|
| `q`            | 是   | -    | 非空       | 查询文本 |
| `topK`         | 否   | 5    | 1..100     | 返回前 K 个 |
| `minScore`     | 否   | 0.0  | -1.0..1.0  | 最低分阈值,两个模式都生效 |
| `columns`      | 否   | -    | 模式二必传 | 逗号分隔列名。空 → 模式一;非空 → 模式二 |
| `docId`        | 否   | -    | -          | 按 docId 过滤候选向量 |

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
