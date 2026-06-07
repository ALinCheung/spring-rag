# spring-rag

本地 RAG 向量检索 demo：Spring Boot 3.3 + Java 17 + ONNX Runtime + DJL HuggingFace tokenizer。
不依赖任何外部向量数据库，文本向量化、存储、检索全在内存里完成。

## 模型
`E:\project_modelscope\bge-small-zh-v1.5\onnx`  
已导出为 ONNX (model.onnx + tokenizer.json)，BGE 中文 512 维 embedding。

## 启动
```bash
mvn spring-boot:run
```
服务端口 `8088`。

## 接口
- `POST /api/rag/upload`  form-data 字段 `file`，上传 utf-8 文本文件
- `GET  /api/rag/search?q=...&topK=5`  相似度检索
- `GET  /api/rag/stats`  当前向量条数

## 设计要点
- 分段：滑动窗口 + 标点切分，配置 `rag.chunk.size` / `rag.chunk.overlap`
- 池化：attention-mask aware mean pooling（与 sentence-transformers 默认行为一致）
- 归一化：L2 normalize，余弦相似度 = 点积
- 存储：进程内 `VectorStore`，读写锁保护，List<Entry> 线性扫描
- 推理：ONNX Runtime 加载 model.onnx；DJL tokenizer 加载 tokenizer.json

## 后续可替换
- 替换 `VectorStore` 为 HNSW / Faiss（需要时再说）
- 加 rerank 模型
- 加文件类型支持（pdf/docx）
