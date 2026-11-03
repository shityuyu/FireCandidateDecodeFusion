#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FireCandidateDecodeFusion 数据生成脚本

功能：
  1. 生成随机测试数据 raw_pred[B,N,6]、priors[N,5]，保存为 .bin 文件。
  2. 用 numpy 实现题目规定的参考实现，生成 expected decoded[B,N,5]，保存为 .bin。

公式（严格按赛题 3.4）：
  sx = sigmoid(tx), sy = sigmoid(ty), sw = sigmoid(tw), sh = sigmoid(th)
  x  = (2*sx - 0.5 + gx) * stride
  y  = (2*sy - 0.5 + gy) * stride
  w  = (2*sw)^2 * anchor_w * stride
  h  = (2*sh)^2 * anchor_h * stride
  obj_score    = sigmoid(obj_logit)
  fire_score_cls = sigmoid(fire_logit)
  fire_score   = obj_score * fire_score_cls
  输出 [x, y, w, h, fire_score]

用法：
  python gen_data.py --batch 4 --candidate 4096 --dtype fp16 --outdir ./data
"""
import argparse
import os

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def reference(raw_pred, priors):
    """参考实现，返回 [B, N, 5] 的 float32 结果。"""
    raw = raw_pred.astype(np.float32)
    tx = raw[..., 0]
    ty = raw[..., 1]
    tw = raw[..., 2]
    th = raw[..., 3]
    obj_logit = raw[..., 4]
    fire_logit = raw[..., 5]

    gx = priors[:, 0]
    gy = priors[:, 1]
    stride = priors[:, 2]
    aw = priors[:, 3]
    ah = priors[:, 4]

    sx = sigmoid(tx)
    sy = sigmoid(ty)
    sw = sigmoid(tw)
    sh = sigmoid(th)

    x = (2.0 * sx - 0.5 + gx) * stride
    y = (2.0 * sy - 0.5 + gy) * stride
    w = np.square(2.0 * sw) * aw * stride
    h = np.square(2.0 * sh) * ah * stride
    fire_score = sigmoid(obj_logit) * sigmoid(fire_logit)

    return np.stack([x, y, w, h, fire_score], axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="B")
    parser.add_argument("--candidate", type=int, default=4096, help="N")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--outdir", type=str, default="./data")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    B, N = args.batch, args.candidate
    rng = np.random.default_rng(2026)

    # logit 使用 [-6, 6] 区间的随机数，覆盖 sigmoid 两端饱和与线性区。
    raw_pred = rng.uniform(-6.0, 6.0, size=(B, N, 6)).astype(np.float32)

    # priors：gx/gy 非负网格坐标；stride/anchor 为正。
    gx = rng.uniform(0.0, 64.0, size=(N,))
    gy = rng.uniform(0.0, 64.0, size=(N,))
    stride = rng.uniform(8.0, 64.0, size=(N,))
    aw = rng.uniform(0.5, 4.0, size=(N,))
    ah = rng.uniform(0.5, 4.0, size=(N,))
    priors = np.stack([gx, gy, stride, aw, ah], axis=-1).astype(np.float32)

    expected = reference(raw_pred, priors)

    if args.dtype == "fp16":
        raw_pred_bin = raw_pred.astype(np.float16)
        expected_bin = expected.astype(np.float16)
    else:
        raw_pred_bin = raw_pred
        expected_bin = expected

    raw_pred_bin.tofile(os.path.join(args.outdir, "raw_pred.bin"))
    priors.tofile(os.path.join(args.outdir, "priors.bin"))
    expected_bin.tofile(os.path.join(args.outdir, "decoded_expected.bin"))

    print(f"[gen_data] B={B} N={N} dtype={args.dtype}")
    print(f"[gen_data] raw_pred -> {args.outdir}/raw_pred.bin  ({raw_pred_bin.nbytes} bytes)")
    print(f"[gen_data] priors   -> {args.outdir}/priors.bin  ({priors.nbytes} bytes)")
    print(f"[gen_data] expected -> {args.outdir}/decoded_expected.bin  ({expected_bin.nbytes} bytes)")


if __name__ == "__main__":
    main()
