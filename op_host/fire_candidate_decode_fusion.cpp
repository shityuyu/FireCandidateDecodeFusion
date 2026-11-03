#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/fire_candidate_decode_fusion_tiling.h"

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    auto rawPredStorage = context->GetInputShape(0);
    auto priorsStorage = context->GetInputShape(1);
    auto rawPredDesc = context->GetInputDesc(0);
    auto priorsDesc = context->GetInputDesc(1);
    if (rawPredStorage == nullptr || priorsStorage == nullptr ||
        rawPredDesc == nullptr || priorsDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape& rawPredShape = rawPredStorage->GetStorageShape();
    const gert::Shape& priorsShape = priorsStorage->GetStorageShape();
    if (rawPredShape.GetDimNum() != 3 || priorsShape.GetDimNum() != 2 ||
        rawPredShape.GetDim(2) != 6 || priorsShape.GetDim(1) != 5) {
        return ge::GRAPH_FAILED;
    }
    if (rawPredShape.GetDim(0) <= 0 || rawPredShape.GetDim(1) <= 0 ||
        rawPredShape.GetDim(1) != priorsShape.GetDim(0)) {
        return ge::GRAPH_FAILED;
    }

    const ge::DataType rawPredType = rawPredDesc->GetDataType();
    if (rawPredType != ge::DT_FLOAT && rawPredType != ge::DT_FLOAT16) {
        return ge::GRAPH_FAILED;
    }
    if (priorsDesc->GetDataType() != ge::DT_FLOAT) {
        return ge::GRAPH_FAILED;
    }

    const uint32_t batchSize = static_cast<uint32_t>(rawPredShape.GetDim(0));
    const uint32_t candidateNum = static_cast<uint32_t>(rawPredShape.GetDim(1));
    const uint32_t dtype =
        (rawPredType == ge::DT_FLOAT) ? FCD_DTYPE_FP32 : FCD_DTYPE_FP16;

    auto ascendcPlatform =
        platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t maxCoreNum = ascendcPlatform.GetCoreNumAiv();
    if (maxCoreNum == 0) {
        maxCoreNum = FCD_FALLBACK_CORE_NUM;
    }
    const uint32_t candidateBlocks =
        (candidateNum + FCD_CANDIDATES_PER_BLOCK - 1) / FCD_CANDIDATES_PER_BLOCK;
    uint32_t coreNum = candidateBlocks;
    if (coreNum == 0) {
        coreNum = 1;
    }
    if (coreNum > maxCoreNum) {
        coreNum = maxCoreNum;
    }
    if (coreNum == 0) {
        return ge::GRAPH_FAILED;
    }

    FireCandidateDecodeFusionTilingData* tiling =
        context->GetTilingData<FireCandidateDecodeFusionTilingData>();
    if (tiling == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling->batchSize = batchSize;
    tiling->candidateNum = candidateNum;
    tiling->dtype = dtype;
    tiling->coreNum = coreNum;

    context->SetBlockDim(coreNum);
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    if (workspaceSize == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaceSize[0] = 0;
    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static graphStatus InferShape(gert::InferShapeContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape* rawPredShape = context->GetInputShape(0);
    if (rawPredShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (rawPredShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    gert::Shape* decodedShape = context->GetOutputShape(0);
    if (decodedShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    decodedShape->SetDimNum(3);
    decodedShape->SetDim(0, rawPredShape->GetDim(0));
    decodedShape->SetDim(1, rawPredShape->GetDim(1));
    decodedShape->SetDim(2, 5);
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class FireCandidateDecodeFusion : public OpDef {
public:
    explicit FireCandidateDecodeFusion(const char* name) : OpDef(name) {
        this->Input("raw_pred")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("priors")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("decoded")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(FireCandidateDecodeFusion);

}  // namespace ops
