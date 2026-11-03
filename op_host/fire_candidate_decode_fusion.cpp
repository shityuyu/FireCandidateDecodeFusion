#include "fire_candidate_decode_fusion_tiling.h"
#include "register/op_def_registry.h"

namespace ge {

// 输出 decoded 的形状：[B, N, 5]，其中 B、N 取自 raw_pred 的前两维。
static ge::graphStatus InferShapeFireCandidateDecodeFusion(gert::InferShapeContext* context) {
    const gert::Shape* rawShape = context->GetInputShape(0);
    if (rawShape == nullptr || rawShape->GetDimNum() != 3) {
        return ge::GRAPH_FAILED;
    }
    uint32_t batchSize = static_cast<uint32_t>(rawShape->GetDim(0));
    uint32_t candidateNum = static_cast<uint32_t>(rawShape->GetDim(1));

    gert::Shape* outShape = context->GetOutputShape(0);
    outShape->SetDimNum(3);
    outShape->SetDim(0, batchSize);
    outShape->SetDim(1, candidateNum);
    outShape->SetDim(2, 5);
    return ge::GRAPH_SUCCESS;
}

// 输出 decoded 的数据类型与 raw_pred 保持一致。
static ge::graphStatus InferDataTypeFireCandidateDecodeFusion(gert::InferDataTypeContext* context) {
    auto rawPredType = context->GetInputDataType(0);
    context->SetOutputDataType(0, rawPredType);
    return ge::GRAPH_SUCCESS;
}

REGISTER_CUSTOM_OP("FireCandidateDecodeFusion")
    .FrameworkType(ONNX)
    .Input("raw_pred")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .Input("priors")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .Output("decoded")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .InferShape(InferShapeFireCandidateDecodeFusion)
    .InferDataType(InferDataTypeFireCandidateDecodeFusion)
    .Tiling(optiling::TilingFuncFireCandidateDecodeFusion);

}  // namespace ge
