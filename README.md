# FireCandidateDecodeFusion 自定义算子

昇腾 Ascend C 实现的 NPU 融合算子：**目标候选框解码与置信度融合**。

对应赛题：CANNJudge「FireCandidateDecodeFusion」。

## 一、算子功能

接收标准化候选预测 `raw_pred` 与先验框 `priors`，依次完成：

```
Sigmoid 激活 → 中心坐标解码 → 宽高解码 → 目标/山火置信度融合 → 候选结果写回
```

| 张量 | 形状 | 类型 | 说明 |
|------|------|------|------|
| raw_pred | [B, N, 6] | fp16 / fp32 | 每候选 `[tx,ty,tw,th,obj_logit,fire_logit]` |
| priors | [N, 5] | fp32 | 每候选 `[gx,gy,stride,anchor_w,anchor_h]` |
| decoded | [B, N, 5] | fp16 / fp32 | 每候选 `[x,y,w,h,fire_score]`，类型同 raw_pred |

### 计算规则（严格按赛题 3.4）

```
sx = sigmoid(tx),  sy = sigmoid(ty),  sw = sigmoid(tw),  sh = sigmoid(th)
x  = (2·sx − 0.5 + gx) · stride
y  = (2·sy − 0.5 + gy) · stride
w  = (2·sw)² · anchor_w · stride
h  = (2·sh)² · anchor_h · stride
fire_score = sigmoid(obj_logit) · sigmoid(fire_logit)
```

输出顺序固定 `[x, y, w, h, fire_score]`，候选顺序不可重排，不做 NMS/TopK/阈值过滤。

## 二、工程结构

```
FireCandidateDecodeFusion/
├── CMakeLists.txt
├── fire_candidate_decode_fusion.h        # aclnn 接口声明
├── fire_candidate_decode_fusion.cpp      # aclnn 接口实现（可省略，见下）
├── op_host/
│   ├── fire_candidate_decode_fusion_tiling.h     # Tiling 数据结构
│   ├── fire_candidate_decode_fusion_tiling.cpp   # TilingFunc（核数/workspace）
│   └── fire_candidate_decode_fusion.cpp          # InferShape + InferDataType + REGISTER_CUSTOM_OP
├── op_kernel/
│   └── fire_candidate_decode_fusion.cpp          # NPU Kernel（Ascend C）
└── scripts/
    ├── gen_data.py        # 生成测试数据 + numpy 参考实现
    └── verify_result.py   # 精度校验
```

## 三、关键实现要点

1. **多核分片**：按候选维度 `n` 分片，每个 AI Core 处理全部 batch 中的一段候选；`priors`
   跨 batch 共享，每核只加载一次。
2. **float32 中间精度**：无论 `raw_pred` 是 fp16 还是 fp32，全部 logit 与先验参数先转
   float32 完成 Sigmoid、解码、平方、乘法、融合；fp16 输入时最后再 `CAST_ROUND` 转回 fp16，
   满足赛题 4.1 的精度要求。
3. **尾块安全**：每块 `BLOCK_N = 256` 个候选，向量化处理 8 对齐部分，剩余不足 8 个的尾块
   走标量路径，正确处理任意 `N`（含非 32/64 整数倍）。
4. **无中间 GM 落地**：输入一次性搬入 UB，计算与重排全部在片上完成，`workspaceSize = 0`。

## 四、编译

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend

cd FireCandidateDecodeFusion
cmake -S . -B build \
      -DCMAKE_SOC_VERSION=Ascend910B \
      -DCMAKE_RUN_MODE=npu
cmake --build build
```

> 不同 CANN 版本的编译器路径 / cmake 工具链略有差异，如遇编译问题请对照官方样例
> `samples/operator/AddCustom` 的 CMakeLists 与构建脚本微调 `KERNEL_CXX` 等变量。

## 五、正确性自测（不依赖 NPU）

用 numpy 验证公式理解与参考实现是否一致，再与 kernel 实际输出比对：

```bash
cd scripts
python gen_data.py --batch 4 --candidate 4096 --dtype fp16 --outdir ./data
python verify_result.py --actual ./data/decoded.bin \
    --expected ./data/decoded_expected.bin --batch 4 --candidate 4096 --dtype fp16
```

精度标准（赛题 4.1）：fp32 用 `atol=rtol=1e-5`；fp16 用 `atol=rtol=3e-3`。

## 六、提交注意事项

- 算子名固定为 `FireCandidateDecodeFusion`。
- `raw_pred` 与 `decoded` 类型一致；`priors` 固定 fp32。
- 若平台用自有 harness 调用已注册算子，`fire_candidate_decode_fusion.cpp`（aclnn 封装）
  可省略，核心是 `op_host` 的 `REGISTER_CUSTOM_OP` + `op_kernel`。
