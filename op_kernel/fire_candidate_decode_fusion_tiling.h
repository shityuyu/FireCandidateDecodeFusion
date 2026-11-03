#ifndef FIRE_CANDIDATE_DECODE_FUSION_TILING_H
#define FIRE_CANDIDATE_DECODE_FUSION_TILING_H

#include <cstdint>

constexpr uint32_t FCD_DTYPE_FP16 = 0;
constexpr uint32_t FCD_DTYPE_FP32 = 1;
constexpr uint32_t FCD_CANDIDATES_PER_BLOCK = 256;
constexpr uint32_t FCD_FALLBACK_CORE_NUM = 20;

struct FireCandidateDecodeFusionTilingData {
    uint32_t batchSize;
    uint32_t candidateNum;
    uint32_t dtype;
    uint32_t coreNum;
};

#endif  // FIRE_CANDIDATE_DECODE_FUSION_TILING_H
