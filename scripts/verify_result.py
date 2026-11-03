#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FireCandidateDecodeFusion 结果校验脚本

将 kernel 实际输出的 decoded.bin 与参考输出 decoded_expected.bin 逐元素比对，
按赛题 4.1 精度要求判断是否通过：
  fp32: atol <= 1e-5 且 rtol <= 1e-5
  fp16: atol <= 3e-3 且 rtol <= 3e-3

用法：
  python verify_result.py --actual ./data/decoded.bin --expected ./data/decoded_expected.bin \
      --batch 4 --candidate 4096 --dtype fp16
"""
import argparse

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", type=str, required=True, help="kernel 输出的 decoded.bin")
    parser.add_argument("--expected", type=str, required=True, help="参考输出 decoded_expected.bin")
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--candidate", type=int, required=True)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], required=True)
    args = parser.parse_args()

    B, N = args.batch, args.candidate
    dt = np.float16 if args.dtype == "fp16" else np.float32

    actual = np.fromfile(args.actual, dtype=dt).reshape(B, N, 5).astype(np.float32)
    expected = np.fromfile(args.expected, dtype=dt).reshape(B, N, 5).astype(np.float32)

    atol = 1e-5 if args.dtype == "fp32" else 3e-3
    rtol = 1e-5 if args.dtype == "fp32" else 3e-3

    diff = np.abs(actual - expected)
    ok = np.allclose(actual, expected, atol=atol, rtol=rtol)

    # 输出每个字段的统计信息
    fields = ["x", "y", "w", "h", "fire_score"]
    print(f"[verify] B={B} N={N} dtype={args.dtype} atol={atol} rtol={rtol}")
    for i, name in enumerate(fields):
        d = diff[..., i]
        print(f"  {name:>10s}: max_abs={d.max():.6e}  mean_abs={d.mean():.6e}")

    if ok:
        print("[verify] PASS")
    else:
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"[verify] FAIL  max diff at {idx}: actual={actual[idx]:.8f} expected={expected[idx]:.8f}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
