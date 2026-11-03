#ifndef FIRE_CANDIDATE_DECODE_FUSION_TILING_H
#define FIRE_CANDIDATE_DECODE_FUSION_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t DTYPE_FP16 = 0;
constexpr uint32_t DTYPE_FP32 = 1;
constexpr uint32_t FCD_CANDIDATES_PER_BLOCK = 256;
constexpr uint32_t FCD_FALLBACK_CORE_NUM = 20;

BEGIN_TILING_DATA_DEF(FireCandidateDecodeFusionTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, candidateNum);
    TILING_DATA_FIELD_DEF(uint32_t, dtype);
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FireCandidateDecodeFusion, FireCandidateDecodeFusionTilingData);

}  // namespace optiling

#endif  // FIRE_CANDIDATE_DECODE_FUSION_TILING_H
