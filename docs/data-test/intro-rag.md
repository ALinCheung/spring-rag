# Spring RAG — 项目简介

## 项目目标 (Project Goals)

构建一个**零外部依赖**、**全本地运行**的 RAG 检索后端，便于研究与演示。
This project builds a **zero-external-dependency**, **fully local** RAG retrieval backend for research and demos.

## 技术栈 (Tech Stack)

- **Spring Boot 3.3** — Java 17 web 框架 (web framework)
- **ONNX Runtime 1.18** — 跨平台模型推理 (cross-platform model inference)
- **DJL HuggingFace Tokenizers 0.30** — 与 Transformers 兼容的 tokenizer
- **Apache Commons CSV 1.11** — RFC 4180 标准 CSV 解析
- **Lombok** — 减少样板代码 (boilerplate reduction)
- **BGE-small-zh-v1.5** — BGE 中文小模型，512 维 embedding

## 模块划分 (Module Layout)

- `embedding/` — ONNX 模型加载与推理
- `service/` — 业务编排：入库、同步、检索
- `store/` — 内存向量库与原始文档仓库
- `sync/` — 启动 / 定时 / 手动三路同步调度
- `controller/` — HTTP 端点与全局异常处理
- `dto/` — 响应实体类 (response DTOs)
- `model/` — 领域模型 (domain model)

## 数据流 (Data Flow)

1. 用户上传文件 → `DocumentRepository` 落盘到 `rag.docs.path`
2. `RagService.ingest` 读取文件 → 分块 → `OnnxBgeEmbeddingService` 向量化 → 入库
3. 检索时把查询文本也向量化 → 与库里向量做点积 (L2 归一化后即余弦相似度)
4. 同步任务会清空当前内存，再扫目录重建 (full-override semantics)
