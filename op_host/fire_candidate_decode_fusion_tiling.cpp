#include "fire_candidate_decode_fusion_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {

constexpr uint32_t BLOCK_N = 256;      // 与 kernel 侧一致
constexpr uint32_t MAX_CORE_NUM = 40;  // 核数上限（910B 单 die 20 核 / 910 32 核，留余量）

static ge::graphStatus TilingFuncFireCandidateDecodeFusion(gert::TilingContext* context) {
    FireCandidateDecodeFusionTilingData tiling;

    // raw_pred: [B, N, 6]
    auto rawPredDesc = context->GetInputDesc(0);
    auto rawPredShape = rawPredDesc->GetShape();
    uint32_t batchSize = static_cast<uint32_t>(rawPredShape.GetDim(0));
    uint32_t candidateNum = static_cast<uint32_t>(rawPredShape.GetDim(1));

    // dtype：raw_pred 与 decoded 类型一致；priors 固定 float32。
    uint32_t dtype = (rawPredDesc->GetDataType() == ge::DT_FLOAT)
                         ? optiling::DTYPE_FP32
                         : optiling::DTYPE_FP16;

    // 核数：按候选块数确定，并夹在 [1, MAX_CORE_NUM]。
    uint32_t coreNum = (candidateNum + BLOCK_N - 1) / BLOCK_N;
    if (coreNum < 1) {
        coreNum = 1;
    }
    if (coreNum > MAX_CORE_NUM) {
        coreNum = MAX_CORE_NUM;
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
