#include "kernel_operator.h"
#include "fire_candidate_decode_fusion_tiling.h"

using namespace AscendC;

__aicore__ inline float FireCandidateSigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <typename T>
class KernelFireCandidateDecodeFusion {
public:
    __aicore__ inline KernelFireCandidateDecodeFusion() {}

    __aicore__ inline void Init(
        GM_ADDR rawPred, GM_ADDR priors, GM_ADDR decoded,
        uint32_t batchSize, uint32_t candidateNum) {
        this->batchSize = batchSize;
        this->candidateNum = candidateNum;
        rawPredGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(rawPred));
        priorsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(priors));
        decodedGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(decoded));
    }

    __aicore__ inline void Process() {
        uint32_t coreNum = GetBlockNum();
        uint32_t coreId = GetBlockIdx();

        // Direct scalar access avoids all DataCopy alignment, tail and pipeline
        // synchronization hazards. Each candidate is owned by exactly one core.
        for (uint32_t b = 0; b < batchSize; ++b) {
            for (uint32_t n = coreId; n < candidateNum; n += coreNum) {
                uint32_t rawOffset = (b * candidateNum + n) * 6;
                uint32_t priorOffset = n * 5;
                uint32_t outOffset = (b * candidateNum + n) * 5;

                float tx = static_cast<float>(rawPredGm.GetValue(rawOffset + 0));
                float ty = static_cast<float>(rawPredGm.GetValue(rawOffset + 1));
                float tw = static_cast<float>(rawPredGm.GetValue(rawOffset + 2));
                float th = static_cast<float>(rawPredGm.GetValue(rawOffset + 3));
                float objLogit = static_cast<float>(rawPredGm.GetValue(rawOffset + 4));
                float fireLogit = static_cast<float>(rawPredGm.GetValue(rawOffset + 5));

                float gx = priorsGm.GetValue(priorOffset + 0);
                float gy = priorsGm.GetValue(priorOffset + 1);
                float stride = priorsGm.GetValue(priorOffset + 2);
                float anchorW = priorsGm.GetValue(priorOffset + 3);
                float anchorH = priorsGm.GetValue(priorOffset + 4);

                float sx = FireCandidateSigmoid(tx);
                float sy = FireCandidateSigmoid(ty);
                float sw = FireCandidateSigmoid(tw);
                float sh = FireCandidateSigmoid(th);

                float x = (2.0f * sx - 0.5f + gx) * stride;
                float y = (2.0f * sy - 0.5f + gy) * stride;
                float w = (2.0f * sw) * (2.0f * sw) * anchorW * stride;
                float h = (2.0f * sh) * (2.0f * sh) * anchorH * stride;
                float score =
                    FireCandidateSigmoid(objLogit) * FireCandidateSigmoid(fireLogit);

                decodedGm.SetValue(outOffset + 0, static_cast<T>(x));
                decodedGm.SetValue(outOffset + 1, static_cast<T>(y));
                decodedGm.SetValue(outOffset + 2, static_cast<T>(w));
                decodedGm.SetValue(outOffset + 3, static_cast<T>(h));
                decodedGm.SetValue(outOffset + 4, static_cast<T>(score));
            }
        }
    }

private:
    GlobalTensor<T> rawPredGm;
    GlobalTensor<float> priorsGm;
    GlobalTensor<T> decodedGm;
    uint32_t batchSize;
    uint32_t candidateNum;
};

extern "C" __global__ __aicore__ void fire_candidate_decode_fusion(
    GM_ADDR rawPred, GM_ADDR priors, GM_ADDR decoded,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(FireCandidateDecodeFusionTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;

    if (tilingData.dtype == FCD_DTYPE_FP16) {
        KernelFireCandidateDecodeFusion<half> op;
        op.Init(rawPred, priors, decoded, tilingData.batchSize, tilingData.candidateNum);
        op.Process();
    } else {
        KernelFireCandidateDecodeFusion<float> op;
        op.Init(rawPred, priors, decoded, tilingData.batchSize, tilingData.candidateNum);
        op.Process();
    }
}
