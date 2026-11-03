#include "fire_candidate_decode_fusion_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t BLOCK_N = 256;          // 与 kernel 侧一致
constexpr uint32_t FALLBACK_CORE_NUM = 20; // 平台查询失败时使用 910B 的安全核数

static ge::graphStatus TilingFuncFireCandidateDecodeFusion(gert::TilingContext* context) {
    FireCandidateDecodeFusionTilingData tiling;

    // raw_pred: [B, N, 6]
    auto rawPredDesc = context->GetInputDesc(0);
    if (rawPredDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto rawPredShape = rawPredDesc->GetShape();
    if (rawPredShape.GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batchSize = static_cast<uint32_t>(rawPredShape.GetDim(0));
    uint32_t candidateNum = static_cast<uint32_t>(rawPredShape.GetDim(1));

    // dtype：raw_pred 与 decoded 类型一致；priors 固定 float32。
    uint32_t dtype = (rawPredDesc->GetDataType() == ge::DT_FLOAT)
                         ? optiling::DTYPE_FP32
                         : optiling::DTYPE_FP16;

    // 查询当前 SoC 可用的 AIV 核数，不能超过平台实际资源。
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (maxCoreNum == 0) {
        maxCoreNum = FALLBACK_CORE_NUM;
    }

    // 核数：按候选块数确定，并夹在 [1, 实际 AIV 核数]。
    uint32_t candidateBlocks = (candidateNum + BLOCK_N - 1) / BLOCK_N;
    uint32_t coreNum = candidateBlocks;
    if (coreNum < 1) {
        coreNum = 1;
    }
    if (coreNum > maxCoreNum) {
        coreNum = maxCoreNum;
    }

    tiling.set_batchSize(batchSize);
    tiling.set_candidateNum(candidateNum);
    tiling.set_dtype(dtype);
    tiling.set_coreNum(coreNum);

    context->SetBlockDim(coreNum);
    // 本算子无中间 GM 落地，不需要 workspace。
    context->SetWorkspaceSize(0);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling
