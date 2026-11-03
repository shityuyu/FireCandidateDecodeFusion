#ifndef ACLNN_FIRE_CANDIDATE_DECODE_FUSION_H
#define ACLNN_FIRE_CANDIDATE_DECODE_FUSION_H

#include "acl/acl.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 计算算子所需 workspace 大小，并返回算子执行器。
 * rawPred : [B, N, 6]  fp16 / fp32
 * priors  : [N, 5]     fp32
 * decoded : [B, N, 5]  fp16 / fp32（与 rawPred 同类型）
 */
aclnnStatus aclnnFireCandidateDecodeFusionGetWorkspaceSize(
    const aclTensor* rawPred, const aclTensor* priors, const aclTensor* decoded,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * 在指定 stream 上执行算子。
 */
aclnnStatus aclnnFireCandidateDecodeFusion(void* workspace, uint64_t workspaceSize,
                                           aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // ACLNN_FIRE_CANDIDATE_DECODE_FUSION_H
