# FireCandidateDecodeFusion

Ascend C implementation of candidate-box decoding and fire-confidence fusion.

## Inputs and output

| Tensor | Shape | Type |
|---|---|---|
| `raw_pred` | `[B, N, 6]` | fp16 / fp32 |
| `priors` | `[N, 5]` | fp32 |
| `decoded` | `[B, N, 5]` | same as `raw_pred` |

`raw_pred` contains `[tx, ty, tw, th, obj_logit, fire_logit]`.
`priors` contains `[gx, gy, stride, anchor_w, anchor_h]`.

The output is `[x, y, w, h, fire_score]` in the original candidate order.

```text
sx = sigmoid(tx)
sy = sigmoid(ty)
sw = sigmoid(tw)
sh = sigmoid(th)

x = (2 * sx - 0.5 + gx) * stride
y = (2 * sy - 0.5 + gy) * stride
w = (2 * sw)^2 * anchor_w * stride
h = (2 * sh)^2 * anchor_h * stride
fire_score = sigmoid(obj_logit) * sigmoid(fire_logit)
```

All intermediate arithmetic uses float32. FP16 inputs are converted to float32
for computation and rounded back only when the final value is written.

## Project layout

```text
.
|-- CMakeLists.txt
|-- op_host
|   |-- CMakeLists.txt
|   `-- fire_candidate_decode_fusion.cpp
|-- op_kernel
|   |-- CMakeLists.txt
|   |-- fire_candidate_decode_fusion.cpp
|   |-- fire_candidate_decode_fusion_tiling.h
|   `-- tiling_key_fire_candidate_decode_fusion.h
`-- scripts
    |-- gen_data.py
    `-- verify_result.py
```

The Host implementation validates input shapes and dtypes, infers the output
shape and dtype, and fills `FireCandidateDecodeFusionTilingData`. The Kernel
uses direct scalar global-memory access and multi-core partitioning. This keeps
the correctness path independent of UB alignment and pipeline synchronization
requirements.

## Local reference utilities

The Python scripts generate reference inputs and verify output files:

```bash
python scripts/gen_data.py --batch 4 --candidate 4096 --dtype fp16 --outdir ./data
python scripts/verify_result.py \
  --actual ./data/decoded.bin \
  --expected ./data/decoded_expected.bin \
  --batch 4 --candidate 4096 --dtype fp16
```
