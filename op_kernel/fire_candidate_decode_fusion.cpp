#include "kernel_operator.h"

using namespace AscendC;

// 每核每次处理的候选数量（块大小）。必须为 8 的整数倍，方便向量化；
// 尾块（不足一块或不足 8 个）由标量路径兜底。
constexpr uint32_t BLOCK_N = 256;

// 标量 sigmoid，供尾块使用
__aicore__ inline float FcdSigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// T 为 raw_pred / decoded 的元素类型：half 或 float。
// priors 与全部中间计算固定使用 float32 精度（题目 4.1 要求）。
template <typename T>
class KernelFireCandidateDecodeFusion {
public:
    __aicore__ inline KernelFireCandidateDecodeFusion() {}

    __aicore__ inline void Init(GM_ADDR rawPred, GM_ADDR priors, GM_ADDR decoded,
                                uint32_t batchSize, uint32_t candidateNum) {
        this->batchSize = batchSize;
        this->candidateNum = candidateNum;

        rawPredGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(rawPred));
        priorsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(priors));
        decodedGm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(decoded));

        // ---- 分配 UB（全部按 BLOCK_N 上限分配）----
        rawT = pipe.AllocTensor<T>(BLOCK_N * 6);       // 原始输入块 [cnt,6]（T 类型）
        rawF32 = pipe.AllocTensor<float>(BLOCK_N * 6); // 转 float32 后的输入块
        priorF32 = pipe.AllocTensor<float>(BLOCK_N * 5); // priors 块 [cnt,5]（float32）
        outF32 = pipe.AllocTensor<float>(BLOCK_N * 5);   // 输出块 [cnt,5]（float32）
        outT = pipe.AllocTensor<T>(BLOCK_N * 5);         // 输出块（T 类型）

        // 分离后的字段向量（各 BLOCK_N 个 float32）
        tx = pipe.AllocTensor<float>(BLOCK_N);
        ty = pipe.AllocTensor<float>(BLOCK_N);
        tw = pipe.AllocTensor<float>(BLOCK_N);
        th = pipe.AllocTensor<float>(BLOCK_N);
        objLogit = pipe.AllocTensor<float>(BLOCK_N);
        fireLogit = pipe.AllocTensor<float>(BLOCK_N);
        gx = pipe.AllocTensor<float>(BLOCK_N);
        gy = pipe.AllocTensor<float>(BLOCK_N);
        strideBuf = pipe.AllocTensor<float>(BLOCK_N);
        aw = pipe.AllocTensor<float>(BLOCK_N);
        ah = pipe.AllocTensor<float>(BLOCK_N);

        // 通用临时 buffer
        t1 = pipe.AllocTensor<float>(BLOCK_N);
        t2 = pipe.AllocTensor<float>(BLOCK_N);
    }

    __aicore__ inline void Process() {
        uint32_t coreNum = GetBlockNum();
        uint32_t coreId = GetBlockIdx();

        // 按候选维度 n 分片，各 core 处理全部 batch 中的一段候选。
        uint32_t perCore = (candidateNum + coreNum - 1) / coreNum;
        uint32_t nStart = coreId * perCore;
        if (nStart >= candidateNum) {
            return;
        }
        uint32_t nEnd = (candidateNum < nStart + perCore) ? candidateNum : (nStart + perCore);

        for (uint32_t n0 = nStart; n0 < nEnd; n0 += BLOCK_N) {
            uint32_t cnt = (nEnd - n0) < BLOCK_N ? (nEnd - n0) : BLOCK_N;
            ProcessBlock(n0, cnt);
        }
    }

private:
    __aicore__ inline void ProcessBlock(uint32_t n0, uint32_t cnt) {
        // 1. priors 在所有 batch 间共享，只搬运 + 分离一次。
        DataCopy(priorF32, priorsGm[n0 * 5], cnt * 5);
        SplitPriors(cnt);

        // 2. 对每个 batch 独立搬运 raw_pred、计算、写回。
        for (uint32_t b = 0; b < batchSize; ++b) {
            LoadRaw(b, n0, cnt);
            SplitRaw(cnt);
            Compute(cnt);
            PackOutput(b, n0, cnt);
        }
    }

    // 搬运 raw_pred[b, n0:n0+cnt, :] 到 UB 并统一成 float32
    __aicore__ inline void LoadRaw(uint32_t b, uint32_t n0, uint32_t cnt) {
        uint32_t offset = (b * candidateNum + n0) * 6;
        if constexpr (std::is_same<T, half>::value) {
            DataCopy(rawT, rawPredGm[offset], cnt * 6);
            Cast(rawF32, rawT, RoundMode::CAST_NONE, cnt * 6);
        } else {
            DataCopy(rawF32, rawPredGm[offset], cnt * 6);
        }
    }

    // 将交错布局 [cnt,5] 的 priors 分离为 5 个连续字段向量
    __aicore__ inline void SplitPriors(uint32_t cnt) {
        for (uint32_t i = 0; i < cnt; ++i) {
            gx.SetValue(i, priorF32.GetValue(i * 5 + 0));
            gy.SetValue(i, priorF32.GetValue(i * 5 + 1));
            strideBuf.SetValue(i, priorF32.GetValue(i * 5 + 2));
            aw.SetValue(i, priorF32.GetValue(i * 5 + 3));
            ah.SetValue(i, priorF32.GetValue(i * 5 + 4));
        }
    }

    // 将交错布局 [cnt,6] 的 raw_pred 分离为 6 个连续字段向量
    __aicore__ inline void SplitRaw(uint32_t cnt) {
        for (uint32_t i = 0; i < cnt; ++i) {
            tx.SetValue(i, rawF32.GetValue(i * 6 + 0));
            ty.SetValue(i, rawF32.GetValue(i * 6 + 1));
            tw.SetValue(i, rawF32.GetValue(i * 6 + 2));
            th.SetValue(i, rawF32.GetValue(i * 6 + 3));
            objLogit.SetValue(i, rawF32.GetValue(i * 6 + 4));
            fireLogit.SetValue(i, rawF32.GetValue(i * 6 + 5));
        }
    }

    // 对一段连续的 float32 向量做 sigmoid，结果写回 dst（原地）
    __aicore__ inline void SigmoidVec(LocalTensor<float> &dst, uint32_t cnt) {
        Muls(t1, dst, -1.0f, cnt);   // t1 = -z
        Exp(t2, t1, cnt);            // t2 = exp(-z)
        Adds(t1, t2, 1.0f, cnt);     // t1 = 1 + exp(-z)
        Reciprocal(dst, t1, cnt);    // dst = 1/(1+exp(-z))
    }

    __aicore__ inline void Compute(uint32_t cnt) {
        uint32_t vecCnt = (cnt / 8) * 8;  // 向量化部分（8 对齐）

        if (vecCnt > 0) {
            // sigmoid 各字段（原地覆盖）
            SigmoidVec(tx, vecCnt);
            SigmoidVec(ty, vecCnt);
            SigmoidVec(tw, vecCnt);
            SigmoidVec(th, vecCnt);
            SigmoidVec(objLogit, vecCnt);
            SigmoidVec(fireLogit, vecCnt);

            // ---- 解码与融合（复用 tx/ty/tw/th/objLogit 存结果）----
            // x = (2*sx - 0.5 + gx) * stride
            Muls(t1, tx, 2.0f, vecCnt);
            Adds(t2, t1, -0.5f, vecCnt);
            Add(t1, t2, gx, vecCnt);
            Mul(tx, t1, strideBuf, vecCnt);

            // y = (2*sy - 0.5 + gy) * stride
            Muls(t1, ty, 2.0f, vecCnt);
            Adds(t2, t1, -0.5f, vecCnt);
            Add(t1, t2, gy, vecCnt);
            Mul(ty, t1, strideBuf, vecCnt);

            // w = (2*sw)^2 * anchor_w * stride
            Muls(t1, tw, 2.0f, vecCnt);
            Mul(t2, t1, t1, vecCnt);
            Mul(t1, t2, aw, vecCnt);
            Mul(tw, t1, strideBuf, vecCnt);

            // h = (2*sh)^2 * anchor_h * stride
            Muls(t1, th, 2.0f, vecCnt);
            Mul(t2, t1, t1, vecCnt);
            Mul(t1, t2, ah, vecCnt);
            Mul(th, t1, strideBuf, vecCnt);

            // fire_score = sigmoid(obj_logit) * sigmoid(fire_logit)
            Mul(objLogit, objLogit, fireLogit, vecCnt);
        }

        // ---- 尾块（不足 8 的部分）标量处理 ----
        for (uint32_t i = vecCnt; i < cnt; ++i) {
            float sx = FcdSigmoid(tx.GetValue(i));
            float sy = FcdSigmoid(ty.GetValue(i));
            float sw = FcdSigmoid(tw.GetValue(i));
            float sh = FcdSigmoid(th.GetValue(i));
            float obj = FcdSigmoid(objLogit.GetValue(i));
            float fire = FcdSigmoid(fireLogit.GetValue(i));

            float gxi = gx.GetValue(i);
            float gyi = gy.GetValue(i);
            float stri = strideBuf.GetValue(i);
            float awi = aw.GetValue(i);
            float ahi = ah.GetValue(i);

            float px = (2.0f * sx - 0.5f + gxi) * stri;
            float py = (2.0f * sy - 0.5f + gyi) * stri;
            float pw = (2.0f * sw) * (2.0f * sw) * awi * stri;
            float ph = (2.0f * sh) * (2.0f * sh) * ahi * stri;
            float score = obj * fire;

            tx.SetValue(i, px);
            ty.SetValue(i, py);
            tw.SetValue(i, pw);
            th.SetValue(i, ph);
            objLogit.SetValue(i, score);
        }
    }

    // 将计算结果 [x,y,w,h,fire_score] 重排回交错布局并写回 GM
    __aicore__ inline void PackOutput(uint32_t b, uint32_t n0, uint32_t cnt) {
        for (uint32_t i = 0; i < cnt; ++i) {
            outF32.SetValue(i * 5 + 0, tx.GetValue(i));
            outF32.SetValue(i * 5 + 1, ty.GetValue(i));
            outF32.SetValue(i * 5 + 2, tw.GetValue(i));
            outF32.SetValue(i * 5 + 3, th.GetValue(i));
            outF32.SetValue(i * 5 + 4, objLogit.GetValue(i));
        }

        uint32_t offset = (b * candidateNum + n0) * 5;
        if constexpr (std::is_same<T, half>::value) {
            Cast(outT, outF32, RoundMode::CAST_ROUND, cnt * 5);
            DataCopy(decodedGm[offset], outT, cnt * 5);
        } else {
            DataCopy(decodedGm[offset], outF32, cnt * 5);
        }
    }

private:
    TPipe pipe;
    GlobalTensor<T> rawPredGm;
    GlobalTensor<float> priorsGm;
    GlobalTensor<T> decodedGm;

    uint32_t batchSize;
    uint32_t candidateNum;

    LocalTensor<T> rawT;
    LocalTensor<T> outT;
    LocalTensor<float> rawF32;
    LocalTensor<float> priorF32;
    LocalTensor<float> outF32;
    LocalTensor<float> tx, ty, tw, th, objLogit, fireLogit;
    LocalTensor<float> gx, gy, strideBuf, aw, ah;
    LocalTensor<float> t1, t2;
};

extern "C" __global__ __aicore__ void fire_candidate_decode_fusion(
    GM_ADDR rawPred, GM_ADDR priors, GM_ADDR decoded,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.dtype == optiling::DTYPE_FP16) {
        KernelFireCandidateDecodeFusion<half> op;
        op.Init(rawPred, priors, decoded, tilingData.batchSize, tilingData.candidateNum);
        op.Process();
    } else {
        KernelFireCandidateDecodeFusion<float> op;
        op.Init(rawPred, priors, decoded, tilingData.batchSize, tilingData.candidateNum);
        op.Process();
    }
}
