# -*- coding: utf-8 -*-
"""
export_to_onnx.py — 把 sentence-transformers 包装的 HF 模型转成单个 model.onnx。

工作流见 ../SKILL.md。

依赖(在 venv 中):torch(CPU), sentence-transformers>=2.7,<3, onnx, onnxruntime。

关键决策:
- 用 STWrapper(nn.Module) 把三个张量拼成 features dict,一次性序列化整条 pipeline
  (Transformer + Pooling + Normalize),而不是只导出 BertModel(optimum-cli 的缺陷)。
- token_type_ids 自适应:有就透传,没有就 torch.zeros_like 填全 0,
  保证 BGE(BERT 家族)与 paraphrase(DistilBERT 家族)通用。
- opset=14、动态 batch/sequence 轴。

典型用法:
    python export_to_onnx.py \\
        --model-dir "E:\\path\\to\\model" \\
        --out-dir   "E:\\path\\to\\model\\onnx"
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer


# ============================================================
# 自检:numpy<2
# ============================================================
def check_numpy_version():
    """钉住 numpy<2,避免 torch 2.3.x ABI 报错。"""
    import numpy as np
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        print(f"[FATAL] 检测到 numpy {np.__version__} (>= 2),与 torch 2.3.x ABI 不兼容。",
              file=sys.stderr)
        print(f"        请先执行:pip install \"numpy<2\"", file=sys.stderr)
        sys.exit(1)


# ============================================================
# STWrapper:接受 3 个独立张量,内部组装 features dict
# ============================================================
class STWrapper(nn.Module):
    """把 SentenceTransformer 包装成 onnx-friendly 的接口。

    输入:input_ids, attention_mask, token_type_ids  (B, L) int64
    输出:sentence_embedding                         (B, dim) float32, L2-normalized
    """

    def __init__(self, st_model: SentenceTransformer):
        super().__init__()
        self.st_model = st_model

    def forward(self, input_ids, attention_mask, token_type_ids):
        features = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            # token_type_ids 始终传入(若调用方给的本来就是全 0,就保持;若没给,
            # 由 export 流程在 features 阶段填全 0),保证 ONNX 输入是 3 张量稳定。
            "token_type_ids": token_type_ids,
        }
        # 内部三个模块按顺序:Transformer -> Pooling -> Normalize
        return self.st_model(features)["sentence_embedding"]


def ensure_token_type_ids(features: dict) -> tuple[dict, bool]:
    """若 features 缺 token_type_ids,用全 0 同形状张量补上。
    返回 (features, synthesized) — synthesized=True 表示是脚本自动填的。"""
    if "token_type_ids" in features:
        return features, False
    input_ids = features["input_ids"]
    features["token_type_ids"] = torch.zeros_like(input_ids, dtype=torch.int64)
    return features, True


# ============================================================
# 复制 tokenizer & ST 配置到 out_dir
# ============================================================
COPY_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
    "special_tokens_map.json",
    "modules.json",
    "1_Pooling",
    "config.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
]


def copy_aux_files(model_dir: Path, out_dir: Path) -> None:
    """把 tokenizer & sentence-transformers 配置从 model_dir 复制到 out_dir。
    sentence-transformers 2.7 缺 save_pretrained,故手动复制。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in COPY_FILES:
        src = model_dir / name
        if not src.exists():
            continue
        dst = out_dir / name
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            copied += 1
        else:
            shutil.copy2(src, dst)
            copied += 1
    print(f"[copy] {copied} 个辅助文件/目录已复制到 {out_dir}")


# ============================================================
# 导出主流程
# ============================================================
def export(model_dir: Path, out_dir: Path, onnx_name: str, opset: int = 14) -> Path:
    print(f"[1/5] 加载 sentence-transformers 模型: {model_dir}")
    model = SentenceTransformer(str(model_dir))
    model.eval()
    print(f"      最大序列长度: {model.max_seq_length}")
    print(f"      模块: {[m.__class__.__name__ for m in model]}")

    print(f"[2/5] 准备 dummy input(tokenizer 真实样本)")
    sample_texts = ["你好,世界"]
    with torch.no_grad():
        features = model.tokenize(sample_texts)
    features, synthesized = ensure_token_type_ids(features)
    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]
    token_type_ids = features["token_type_ids"]
    print(f"      input_ids:      {tuple(input_ids.shape)}")
    print(f"      attention_mask: {tuple(attention_mask.shape)}")
    print(f"      token_type_ids: {tuple(token_type_ids.shape)} (synthesized={synthesized})")

    print(f"[3/5] 配置 dynamic_axes & opset={opset}")
    input_names = ["input_ids", "attention_mask", "token_type_ids"]
    output_names = ["sentence_embedding"]
    dynamic_axes = {
        "input_ids":          {0: "batch", 1: "sequence"},
        "attention_mask":     {0: "batch", 1: "sequence"},
        "token_type_ids":     {0: "batch", 1: "sequence"},
        "sentence_embedding": {0: "batch"},
    }

    wrapper = STWrapper(model)
    wrapper.eval()

    # 冒烟:用 wrapper 跑一次,确认包装不破坏前向
    print(f"[4/5] 冒烟测试 wrapper 跑一次")
    with torch.no_grad():
        out = wrapper(input_ids, attention_mask, token_type_ids)
    print(f"      形状: {tuple(out.shape)}")

    onnx_path = out_dir / onnx_name
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # 清理 out_dir 中已有文件(防止 ModelScope 仓库预生成的 model_O*.onnx /
    # model_qint8_*.onnx 污染产物目录)。白名单文件稍后由 copy_aux_files 重新写入。
    if onnx_path.parent.exists():
        for f in onnx_path.parent.iterdir():
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
    print(f"[5/5] 调用 torch.onnx.export -> {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, token_type_ids),
            str(onnx_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            verbose=False,
        )

    # 复制 tokenizer & ST 配置
    copy_aux_files(model_dir, onnx_path.parent)

    # 列出最终产物
    print(f"\n产物清单:")
    for p in sorted(onnx_path.parent.iterdir()):
        size = p.stat().st_size
        print(f"    {p.name:40s} {size:>12,d} bytes")
    return onnx_path


# ============================================================
# 入口
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="把 sentence-transformers 模型导出为单个 model.onnx(包含 Pooling+Normalize)。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", required=True,
                   help="本地 sentence-transformers 模型目录路径")
    p.add_argument("--out-dir",
                   help="ONNX 产物输出目录(默认 {model-dir}/onnx)")
    p.add_argument("--onnx-name", default="model.onnx",
                   help="ONNX 文件名(默认 model.onnx)")
    p.add_argument("--opset", type=int, default=14,
                   help="ONNX opset 版本(默认 14,兼容 onnxruntime 1.17+)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    check_numpy_version()

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"[FATAL] 模型目录不存在: {model_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else model_dir / "onnx"

    t0 = time.time()
    onnx_path = export(model_dir, out_dir, args.onnx_name, opset=args.opset)
    dt = time.time() - t0
    print(f"\n[OK] ONNX 已生成: {onnx_path}  耗时={dt:.1f}s")


if __name__ == "__main__":
    main()
