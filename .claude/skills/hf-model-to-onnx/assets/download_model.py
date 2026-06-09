# -*- coding: utf-8 -*-
"""
download_model.py — 从 ModelScope 镜像(默认)或 HuggingFace Hub 拉取模型仓库。

工作流见 ../SKILL.md。

依赖:modelscope(>=1.9), huggingface_hub(>=0.20);首次运行前 `pip install -U modelscope huggingface_hub`。

典型用法:
    # 仅下载
    python download_model.py --model-name "sentence-transformers/paraphrase-MiniLM-L3-v2" \
        --output-dir "E:\\project_github\\spring-rag\\src\\main\\resources\\model" \
        --source modelscope

    # 一步端到端(下载 + 转 ONNX + 验证)
    python download_model.py --model-name "sentence-transformers/paraphrase-MiniLM-L3-v2" \
        --output-dir "E:\\project_github\\spring-rag\\src\\main\\resources\\model" \
        --auto-convert

    # 模型已下载,只重新跑 convert(跳过下载)
    python download_model.py --model-name "..." --output-dir "..." --auto-convert --skip-download
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


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
# 下载函数
# ============================================================
def download_via_modelscope(model_id: str, output_dir: Path) -> str:
    """从 ModelScope 拉取。返回落盘的模型目录路径。"""
    from modelscope import snapshot_download
    print(f"  [modelscope] snapshot_download({model_id!r}, cache_dir={output_dir!r})")
    # ModelScope 的 cache_dir 是缓存根目录,会在其下创建模型子目录
    return snapshot_download(model_id, cache_dir=str(output_dir))


def download_via_hf(repo_id: str, output_dir: Path) -> str:
    """从 HuggingFace Hub 拉取。返回落盘的模型目录路径。"""
    from huggingface_hub import snapshot_download
    print(f"  [hf] snapshot_download(repo_id={repo_id!r}, local_dir={output_dir!r})")
    return snapshot_download(repo_id=repo_id, local_dir=str(output_dir))


def try_download(model_id: str, source: str, output_dir: Path) -> tuple[bool, str]:
    """
    尝试下载,返回 (success, model_path)。
    - source == 'modelscope': 走 modelscope
    - source == 'hf': 走 HF
    - 任何异常都返回 (False, str(e))
    """
    try:
        if source == "modelscope":
            path = download_via_modelscope(model_id, output_dir)
        elif source == "hf":
            path = download_via_hf(model_id, output_dir)
        else:
            return False, f"未知 source: {source}"
        return True, path
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# ============================================================
# 三层回退
# ============================================================
def download_with_fallback(
    model_name: str,
    output_dir: Path,
    source: str,
    ms_name: str | None,
    hf_name: str | None,
) -> tuple[str, str]:
    """
    返回 (final_model_path, actual_source_used)。
    优先级:
        1. 用户显式 --ms-name / --hf-name(若 source 与之匹配则用之,否则按 source 走)
        2. 默认 source(modelname 原名)
        3. ModelScope 失败回退小写 + HF
        4. 全失败 raise
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir = output_dir / model_name
    if final_model_dir.exists() and any(final_model_dir.iterdir()):
        print(f"[SKIP] 模型目录已存在且非空: {final_model_dir}")
        return str(final_model_dir), "cache(existing)"

    # === 显式 --ms-name 优先(用于 ModelScope 命名习惯差异) ===
    if source == "modelscope" and ms_name:
        print(f"[1/3] 尝试用户指定的 ModelScope 仓库: {ms_name}")
        ok, info = try_download(ms_name, "modelscope", output_dir)
        if ok:
            return info, f"modelscope(explicit:{ms_name})"
        print(f"        失败: {info}")

    # === 默认 source,用 model_name 原名 ===
    print(f"[2/3] 尝试 {source} 原名: {model_name}")
    ok, info = try_download(model_name, source, output_dir)
    if ok:
        return info, f"{source}(name:{model_name})"
    print(f"        失败: {info}")

    # === ModelScope 失败时小写重试一次(常见命名习惯) ===
    if source == "modelscope":
        lowered = model_name.lower()
        if lowered != model_name:
            print(f"[3/3] 尝试小写: {lowered}")
            ok, info = try_download(lowered, "modelscope", output_dir)
            if ok:
                return info, f"modelscope(lowered:{lowered})"
            print(f"        失败: {info}")

    # === HF 兜底 ===
    hf_repo = hf_name or model_name
    print(f"[fallback] 尝试 HuggingFace: {hf_repo}")
    ok, info = try_download(hf_repo, "hf", output_dir)
    if ok:
        return info, f"hf(fallback:{hf_repo})"
    raise RuntimeError(
        f"所有下载源均失败。最后一次错误: {info}\n"
        f"请确认模型名是否正确,或手动指定 --ms-name / --hf-name"
    )


# ============================================================
# --auto-convert:子进程调用 export + verify
# ============================================================
def run_subprocess(cmd: list[str], cwd: Path) -> int:
    """运行子进程,实时输出。返回 exit code。"""
    print(f"  [subprocess] {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def auto_convert(model_path: str, output_dir: Path, english: bool = False) -> None:
    """下载成功后,调 export + verify。"""
    model_path_p = Path(model_path)
    onnx_dir = model_path_p / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    assets_dir = Path(__file__).parent

    # 1) 导出 ONNX
    print("\n[auto-convert] Step 1/2: export_to_onnx.py")
    cmd_export = [
        sys.executable,
        str(assets_dir / "export_to_onnx.py"),
        "--model-dir", str(model_path_p),
        "--out-dir", str(onnx_dir),
    ]
    rc = run_subprocess(cmd_export, cwd=assets_dir)
    if rc != 0:
        raise RuntimeError(f"export_to_onnx.py 退出码 {rc}")

    # 2) 验证
    print("\n[auto-convert] Step 2/2: verify_onnx.py")
    cmd_verify = [
        sys.executable,
        str(assets_dir / "verify_onnx.py"),
        "--onnx-dir", str(onnx_dir),
        "--model-dir", str(model_path_p),
    ]
    if english:
        cmd_verify.append("--english")
    rc = run_subprocess(cmd_verify, cwd=assets_dir)
    if rc != 0:
        raise RuntimeError(f"verify_onnx.py 退出码 {rc}")

    print("\n[ALL CHECKS PASSED] 下载 + 转换 + 验证 全部成功")


# ============================================================
# 入口
# ============================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="从 ModelScope(默认)或 HuggingFace Hub 拉取 sentence-transformers / HF 模型仓库,可选自动转 ONNX。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-name", required=True,
                   help="HuggingFace 风格的仓库 id,如 'sentence-transformers/paraphrase-MiniLM-L3-v2'")
    p.add_argument("--output-dir", required=True,
                   help="模型落盘的父目录(模型会落在 {output-dir}/{model-name}/ 下)")
    p.add_argument("--source", choices=["modelscope", "hf"], default="modelscope",
                   help="主下载源;失败时自动回退到 HuggingFace Hub")
    p.add_argument("--ms-name", default=None,
                   help="手动指定 ModelScope 仓库 id(覆盖自动解析,常用于命名习惯差异)")
    p.add_argument("--hf-name", default=None,
                   help="手动指定 HuggingFace 仓库 id(覆盖自动解析)")
    p.add_argument("--auto-convert", action="store_true",
                   help="下载成功后自动调 export_to_onnx.py + verify_onnx.py 跑端到端")
    p.add_argument("--english", action="store_true",
                   help="auto-convert 时,verify 用英文默认样本(适用于 paraphrase 等英文模型)")
    p.add_argument("--skip-download", action="store_true",
                   help="跳过下载,直接进入 auto-convert(用于模型已落盘,只想重跑 convert 的场景)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    check_numpy_version()

    output_dir = Path(args.output_dir).resolve()
    if not args.skip_download:
        print(f"[start] model={args.model_name} output={output_dir} source={args.source}")
        t0 = time.time()
        model_path, actual_source = download_with_fallback(
            model_name=args.model_name,
            output_dir=output_dir,
            source=args.source,
            ms_name=args.ms_name,
            hf_name=args.hf_name,
        )
        dt = time.time() - t0
        print(f"[done] 来源={actual_source}")
        print(f"       落盘={model_path}  耗时={dt:.1f}s")
    else:
        model_path = str(output_dir / args.model_name)
        if not Path(model_path).exists():
            print(f"[FATAL] --skip-download 但模型目录不存在: {model_path}", file=sys.stderr)
            sys.exit(1)
        print(f"[skip-download] 复用现有模型目录: {model_path}")

    if args.auto_convert:
        auto_convert(model_path, output_dir, english=args.english)


if __name__ == "__main__":
    main()
