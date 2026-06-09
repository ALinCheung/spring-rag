# -*- coding: utf-8 -*-
"""
verify_onnx.py — 用 onnxruntime 端到端验证 model.onnx。

工作流见 ../SKILL.md。

依赖:onnxruntime, transformers, numpy, sentence-transformers, torch。

关键检查:
1) 输出形状 == (n, embedding_dim)
2) 每行 L2 范数 ≈ 1.0(误差 < 1e-3)
3) 与 PyTorch 路径余弦相似度 ≥ 0.999(仅在 --model-dir 提供时)

典型用法:
    # 中文模型(BGE)
    python verify_onnx.py \\
        --onnx-dir "E:\\path\\to\\onnx" \\
        --model-dir "E:\\path\\to\\model"

    # 英文模型(paraphrase-MiniLM)
    python verify_onnx.py \\
        --onnx-dir "E:\\path\\to\\onnx" \\
        --model-dir "E:\\path\\to\\model" \\
        --english
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")  # 屏蔽 torch/transformers 的非致命 warning


# ============================================================
# 默认样本
# ============================================================
DEFAULT_ZH_TEXTS = [
    "你好,世界",
    "今天天气真不错,适合出门散步",
    "BGE 是一个优秀的中文嵌入模型",
]
DEFAULT_ZH_CROSS_CHECK = [
    "我喜欢在周末去公园散步,呼吸新鲜空气。",
    "深度学习模型可以将文本转换为向量表示。",
    "明天可能会下雨,记得带伞。",
]
DEFAULT_EN_TEXTS = [
    "Hello, world",
    "The weather is great today, perfect for a walk in the park.",
    "ONNX is a portable format for machine learning models.",
]
DEFAULT_EN_CROSS_CHECK = [
    "Machine translation has improved significantly with neural networks.",
    "I love reading books about artificial intelligence and its applications.",
    "Vector databases enable efficient similarity search at scale.",
]


# ============================================================
# 自检
# ============================================================
def check_numpy_version():
    import numpy as np
    major = int(np.__version__.split(".")[0])
    if major >= 2:
        print(f"[FATAL] 检测到 numpy {np.__version__} (>= 2),与 torch 2.3.x ABI 不兼容。",
              file=sys.stderr)
        print(f"        请先执行:pip install \"numpy<2\"", file=sys.stderr)
        sys.exit(1)


def has_normalize_layer(onnx_dir: Path) -> bool:
    """通过 modules.json 判断 sentence-transformers pipeline 是否含 Normalize 层。
    含 → L2 范数 ≈ 1.0;不含 → 范数不归一化(跳过 1.0 检查,只验证有效性)。"""
    modules_json = onnx_dir / "modules.json"
    if not modules_json.exists():
        return True  # 保守:无法判断时按有 Normalize 处理
    try:
        import json
        modules = json.loads(modules_json.read_text(encoding="utf-8"))
        return any("Normalize" in (m.get("type") or "") for m in modules)
    except Exception:
        return True


# ============================================================
# 主流程
# ============================================================
def run_verification(
    onnx_dir: Path,
    model_dir: Path | None,
    test_texts: list[str],
    cross_check_texts: list[str],
) -> int:
    print(f"[1/4] 加载 tokenizer: {onnx_dir}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(onnx_dir))
    print(f"      tokenizer type: {type(tokenizer).__name__}")

    onnx_path = onnx_dir / "model.onnx"
    if not onnx_path.exists():
        print(f"[FATAL] 找不到 {onnx_path}", file=sys.stderr)
        return 1
    print(f"[2/4] 加载 onnxruntime session: {onnx_path}")
    import onnxruntime as ort
    sess = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    print(f"      inputs:  {[i.name for i in sess.get_inputs()]}")
    print(f"      outputs: {[o.name for o in sess.get_outputs()]}")
    for inp in sess.get_inputs():
        print(f"      in  {inp.name:20s} {inp.shape}  {inp.type}")

    print(f"[3/4] 跑推理(共 {len(test_texts)} 条)")
    enc = tokenizer(
        test_texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="np",
    )
    ort_inputs = {
        "input_ids":      enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
        "token_type_ids": enc["token_type_ids"].astype(np.int64),
    }
    for k, v in ort_inputs.items():
        print(f"      {k:20s} {v.shape}  {v.dtype}")

    ort_emb = sess.run(None, ort_inputs)[0]
    print(f"      -> 输出形状: {ort_emb.shape}  dtype: {ort_emb.dtype}")

    print(f"[4/4] 自检")
    # 1) 形状
    emb_dim = ort_emb.shape[1]
    assert ort_emb.shape == (len(test_texts), emb_dim), \
        f"形状错误,期望 ({len(test_texts)}, {emb_dim}), 实际 {ort_emb.shape}"
    print(f"      [OK] 形状 == (n, {emb_dim})")

    # 2) L2 范数 — 仅在模型有 Normalize 层时检查 ≈ 1.0;
    #    无 Normalize(paraphrase 等)时,只检查"非全零、非 NaN/Inf"
    norms = np.linalg.norm(ort_emb, axis=1)
    print(f"      L2 范数: {norms}")
    if has_normalize_layer(onnx_dir):
        assert np.all(np.abs(norms - 1.0) < 1e-3), \
            f"L2 范数偏离 1.0 超过 1e-3,实际 {norms}"
        print(f"      [OK] L2 范数 ≈ 1.0(误差 < 1e-3,模型含 Normalize 层)")
    else:
        assert np.all(np.isfinite(ort_emb)), "ONNX 输出含 NaN/Inf"
        assert np.all(norms > 0), "ONNX 输出全 0"
        assert not np.allclose(ort_emb[0], ort_emb[1]), "ONNX 对不同输入产生相同输出"
        print(f"      [OK] 输出有效(L2 范数 > 0,非 NaN/Inf,样本有区分度;模型无 Normalize 层,跳过 ≈ 1.0 检查)")

    # 3) 抽样
    print(f"      首句向量前 5 维: {ort_emb[0, :5]}")

    # 4) 与 PyTorch 路径的余弦相似度
    if model_dir is not None and model_dir.exists():
        print()
        print(f"[extra] 与 PyTorch 路径对比余弦相似度(model-dir={model_dir})")
        from sentence_transformers import SentenceTransformer
        import torch

        st = SentenceTransformer(str(model_dir))
        st.eval()
        with torch.no_grad():
            pt_emb = st.encode(cross_check_texts, convert_to_numpy=True,
                               normalize_embeddings=False)  # 模型内已有 Normalize

        enc_cc = tokenizer(cross_check_texts, padding=True, truncation=True,
                           max_length=512, return_tensors="np")
        ort_emb_cc = sess.run(None, {
            "input_ids":      enc_cc["input_ids"].astype(np.int64),
            "attention_mask": enc_cc["attention_mask"].astype(np.int64),
            "token_type_ids": enc_cc["token_type_ids"].astype(np.int64),
        })[0]
        # 先 L2 归一化再算内积,得到**真余弦**(无论模型有无 Normalize 都对)
        pt_norm = pt_emb / np.linalg.norm(pt_emb, axis=1, keepdims=True)
        ort_norm = ort_emb_cc / np.linalg.norm(ort_emb_cc, axis=1, keepdims=True)
        cos = np.sum(pt_norm * ort_norm, axis=1)
        print(f"      cross-check 余弦 per-sample: {cos}")
        print(f"      最小值: {cos.min():.6f}")
        assert cos.min() > 0.999, \
            f"ONNX 与 PyTorch 不一致,最小余弦 {cos.min()}"
        print(f"      [OK] cross-check 余弦 > 0.999,ONNX 与 PyTorch 数值一致")
    else:
        print(f"      [skip] 未提供 --model-dir,跳过 PyTorch 交叉校验")

    # 5) 句间语义区分度
    sims = ort_emb @ ort_emb.T
    print(f"\n      ONNX 嵌入矩阵(自身余弦相似度):")
    for i, row in enumerate(sims):
        print(f"        句 {i}: {[f'{x:+.4f}' for x in row]}")

    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="用 onnxruntime 验证导出的 model.onnx,包括形状/L2/与 PyTorch 余弦相似度检查。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx-dir", required=True,
                   help="包含 model.onnx 的目录(也应含 tokenizer.json 等)")
    p.add_argument("--model-dir",
                   help="(可选)原始 sentence-transformers 模型目录,提供时做 PyTorch 路径余弦交叉校验")
    p.add_argument("--test-texts", nargs="+",
                   help="用于 ONNX 推理的测试句子(空格分隔),默认内置 3 条(中文或英文,取决于 --english)")
    p.add_argument("--cross-check-texts", nargs="+",
                   help="用于 PyTorch 交叉校验的句子,默认内置 3 条(中文或英文)")
    p.add_argument("--english", action="store_true",
                   help="使用英文默认样本(适用于 paraphrase 等英文模型);不传则用中文默认")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    check_numpy_version()

    onnx_dir = Path(args.onnx_dir).resolve()
    if not onnx_dir.exists():
        print(f"[FATAL] onnx-dir 不存在: {onnx_dir}", file=sys.stderr)
        return 1

    model_dir = Path(args.model_dir).resolve() if args.model_dir else None

    if args.english:
        test_texts = args.test_texts or DEFAULT_EN_TEXTS
        cross_texts = args.cross_check_texts or DEFAULT_EN_CROSS_CHECK
    else:
        test_texts = args.test_texts or DEFAULT_ZH_TEXTS
        cross_texts = args.cross_check_texts or DEFAULT_ZH_CROSS_CHECK

    print(f"[start] onnx-dir={onnx_dir}  english={args.english}")
    rc = run_verification(onnx_dir, model_dir, test_texts, cross_texts)
    if rc == 0:
        print()
        print("=" * 60)
        print("[ALL CHECKS PASSED] ONNX 模型验证通过,可用于生产部署。")
        print("=" * 60)
    return rc


if __name__ == "__main__":
    sys.exit(main())
