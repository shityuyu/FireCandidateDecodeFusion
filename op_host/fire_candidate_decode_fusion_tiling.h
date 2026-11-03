#ifndef FIRE_CANDIDATE_DECODE_FUSION_TILING_H
#define FIRE_CANDIDATE_DECODE_FUSION_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

// dtype 枚举：与 aclDataType 保持一致
constexpr uint32_t DTYPE_FP16 = 0;  // aclDataType::ACL_FLOAT16
constexpr uint32_t DTYPE_FP32 = 1;  // aclDataType::ACL_FLOAT

BEGIN_TILING_DATA_DEF(FireCandidateDecodeFusionTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);    // B ∈ [1, 8]
    TILING_DATA_FIELD_DEF(uint32_t, candidateNum); // N ∈ [1, 32768]
    TILING_DATA_FIELD_DEF(uint32_t, dtype);        // 0: fp16, 1: fp32
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);      // 实际使用的 AI Core 数
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FireCandidateDecodeFusion, FireCandidateDecodeFusionTilingData);

}  // namespace optiling

#endif  // FIRE_CANDIDATE_DECODE_FUSION_TILING_H
