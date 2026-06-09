---
name: hf-model-to-onnx
description: 把 HuggingFace / sentence-transformers 嵌入模型(BGE、m3e、bce、paraphrase-MiniLM 等)下载并转成单个 ONNX 文件,产出可在 ONNX Runtime 直接调用的归一化句向量。当用户提供 HF 仓库名或本地模型目录、需要部署到 Java/Python/任何 ONNX Runtime 运行时、或要复用已验证的转换参数集(opset=14、动态 batch/sequence、Pooling+Normalize 必入图)时,使用本 skill。
license: MIT
compatibility: Requires Python 3.10+ venv with torch (CPU), sentence-transformers, optimum[exporters], onnx, onnxruntime, transformers, tokenizers, huggingface_hub, modelscope. Windows / Linux / macOS 通用;CPU 即可完成全部工作。
metadata:
  author: hf-model-to-onnx
  version: "1.0"
  generatedBy: "OpenSpec hf-model-to-onnx change"
---

# hf-model-to-onnx

## 适用场景

- 用户给一个 HF 仓库名(如 `sentence-transformers/paraphrase-MiniLM-L3-v2`、`BAAI/bge-small-zh-v1.5`、`BAAI/bge-large-zh-v1.5`),
  需要可被 ONNX Runtime 推理的 `model.onnx` 文件;
- 用户已有本地 sentence-transformers 模型目录,想转成 ONNX 用于 Java/spring-rag / Python 服务部署;
- 需要复用已验证的转换参数集(opset=14、动态 batch/sequence、Pooling+Normalize 必入图、L2 归一化嵌入)。

## 工作流总览(7 步)

```
① 建 venv                →  py -3 -m venv .venv-hf-onnx
② 装依赖(钉 numpy<2)    →  pip install -U ...(见必读约束)
③ 下载模型               →  python assets/download_model.py --model-name <hf_repo> --output-dir <dir> [--auto-convert]
④ 转 ONNX(下载自动 / 手动) →  python assets/export_to_onnx.py --model-dir <path> --out-dir <path>
⑤ 复制 tokenizer & ST 配置(脚本内自动) → 由 export_to_onnx.py 完成后写入 onnx/ 目录
⑥ 验证                   →  python assets/verify_onnx.py --onnx-dir <path> [--model-dir <path>] [--english]
⑦ 报告产出路径            →  onnx/model.onnx,可被 Java spring-rag / Python ONNX Runtime 直接加载
```

## 输入 / 输出契约

```
输入(input_ids, attention_mask, token_type_ids)  — 3 个 int64 张量,各 shape (B, L)
输出(sentence_embedding)                          — 1 个 float32 张量,shape (B, embedding_dim)
```

- `B` 动态(batch size,任意正整数);
- `L` 动态(sequence length,<= 512,会被 tokenizer 自动 padding / truncation);
- `embedding_dim` 由源模型决定(BGE-small-zh = 512、paraphrase-MiniLM-L3-v2 = 384、BGE-large-zh = 1024);
- 输出**已 L2 归一化**(每行范数 ≈ 1.0),可直接做余弦相似度检索;
- **特别说明**:`token_type_ids` 即使源模型无(如 DistilBERT 家族的 paraphrase),
  也由 `STWrapper` 自动填全 0,保持 3 输入接口稳定,部署侧无需写两套加载代码。

## 必读约束(踩过的坑)

1. **`numpy<2` 是硬性约束** —— `torch 2.3.1` 是 numpy 1.x ABI 编译的,碰 numpy 2.x 会触发
   `_ARRAY_API not found` 警告并在 tensor<->numpy 转换时崩溃。装依赖时第一步就 `pip install "numpy<2"`。
   一旦 `torch >= 2.4`,此约束可放开。
2. **动态 batch / sequence 轴** —— `dynamic_axes` 在 export 时必须同时声明 batch 和 sequence 两个轴;
   否则生产侧不能变 batch。
3. **Pooling + Normalize 必入图** —— 不要用 `optimum-cli export onnx --task feature-extraction`,
   那样只导出底层 BertModel,不含 Pooling(CLS)与 Normalize(L2)。本 skill 用 `STWrapper` 把
   整条 pipeline 一次性序列化。
4. **纯中文 / 纯英文样本做数值校验** —— BGE 在中英混排样本上 ONNX 与 PyTorch 余弦约 0.93
   (tokenizer 路径差异),这是 BGE 已知行为,不是 ONNX 错误。验证时用 3 句纯中文或纯英文。
5. **CPU 即可** —— onnxruntime 选 `CPUExecutionProvider`;若用户机器有 GPU,自行改
   `providers=["CUDAExecutionProvider", "CPUExecutionProvider"]`。

## 调用样例

### 样例 A:仅下载(从 ModelScope 镜像)

```bash
python assets/download_model.py \
  --model-name "sentence-transformers/paraphrase-MiniLM-L3-v2" \
  --output-dir "E:\project_github\spring-rag\src\main\resources\model" \
  --source modelscope
```

### 样例 B:一条命令端到端(下载 + 转 ONNX + 验证)

```bash
python assets/download_model.py \
  --model-name "sentence-transformers/paraphrase-MiniLM-L3-v2" \
  --output-dir "E:\project_github\spring-rag\src\main\resources\model" \
  --auto-convert
```

成功后产出:
- 源模型:`<output-dir>/<model-name>/`
- ONNX:`<output-dir>/<model-name>/onnx/model.onnx`
- 报告:`[ALL CHECKS PASSED]`

### 样例 C:模型已下载,只转 + 校验

```bash
# 转换
python assets/export_to_onnx.py \
  --model-dir "E:\project_modelscope\bge-small-zh-v1.5" \
  --out-dir "E:\project_modelscope\bge-small-zh-v1.5\onnx"

# 验证(英文模型加 --english,中文模型省略)
python assets/verify_onnx.py \
  --onnx-dir "E:\project_modelscope\bge-small-zh-v1.5\onnx" \
  --model-dir "E:\project_modelscope\bge-small-zh-v1.5"
```

## 已知陷阱

- **中英混排样本余弦 ≈ 0.93** —— 不是 ONNX 错误,是 BGE tokenizer 在中英混排上的已知行为。
- **Windows 路径含空格** —— 用引号包裹,例如 `--model-dir "C:\path with space\model"`。
- **ModelScope 仓库 ID 大小写** —— ModelScope 镜像常用小写命名(如
  `AI-ModelScope/sentence-transformers-paraphrase-minilm-l3-v2`),
  download_model.py 实施三层回退:原名 → 小写重试 → HF 兜底。
- **torch 2.3.1 + numpy 2.x** —— 见必读约束第 1 条。

## 放置位置建议

- **Java / Spring 项目**:推荐 `<project>/src/main/resources/model/<model-name>/`,
  ONNX 在 `<model-name>/onnx/model.onnx`;运行时由 classloader 加载;
  **务必在项目 `.gitignore` 中加入** `src/main/resources/model/`,避免 jar 体积膨胀。
- **Python 服务**:推荐 `<project>/models/<model-name>/onnx/model.onnx`。
- **跨盘 OK**:skill 不假设模型与 skill 同盘,Windows D:/E:/ 任意。

## 扩展指引

非 BERT/DistilBERT/RoBERTa 家族模型(MDEBERTa、Qwen-Embedding、E5、bge-m3)可能:
- 不需要 `token_type_ids` → `STWrapper` 已自动填全 0,无需改;
- tokenizer 输出 `input_ids` / `attention_mask` 之外还有其他键 → 在 `export_to_onnx.py`
  的 `STWrapper.forward` 中添加对应张量,并把 `input_names` / `dynamic_axes` 同步扩展;
- 嵌入维度 > 1024 → 仍按 `sentence_embedding` 输出,`embedding_dim` 在
  `verify_onnx.py` 中通过 `ort_emb.shape[1]` 自动获取。

详细决策见 OpenSpec change 设计文档:
`openspec/changes/hf-model-to-onnx/design.md`(本仓库内)。
