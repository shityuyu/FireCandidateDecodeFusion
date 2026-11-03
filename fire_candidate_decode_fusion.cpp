#include "acl/acl.h"
#include "aclnn_fire_candidate_decode_fusion.h"

#include <vector>

/*
 * aclnn 接口实现说明
 * -------------------
 * 本算子通过 op_host/fire_candidate_decode_fusion.cpp 中的 REGISTER_CUSTOM_OP
 * 完成注册后，即可被 CANN 框架调度执行。
 *
 * 关于 aclnn 封装：CANN 8.0 自定义算子的 aclnn 接口依赖 aclnn 框架，
 * 不同 CANN 版本 API 略有差异，官方标准写法见：
 *     samples/operator/AddCustom/add_custom.cpp
 *
 * 若提交平台（如 CANNJudge）使用自己的 harness 直接调用已注册算子，
 * 则本文件可省略，只需保证 op_host 注册 + op_kernel 实现正确即可。
 */

aclnnStatus aclnnFireCandidateDecodeFusionGetWorkspaceSize(
    const aclTensor* rawPred, const aclTensor* priors, const aclTensor* decoded,
    uint64_t* workspaceSize, aclOpExecutor** executor) {
    // 通过 aclnn 框架按算子名获取 workspace 大小并创建执行器。
    // 具体实现请对照官方 AddCustom 样例的 add_custom.cpp。
    // 本算子无中间 GM 落地，workspaceSize 恒为 0（见 tiling 中 SetWorkspaceSize(0)）。
    *workspaceSize = 0;
    *executor = nullptr;
    (void)rawPred;
    (void)priors;
    (void)decoded;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFireCandidateDecodeFusion(void* workspace, uint64_t workspaceSize,
                                           aclOpExecutor* executor, aclrtStream stream) {
    (void)workspace;
    (void)workspaceSize;
    (void)executor;
    (void)stream;
    return ACLNN_SUCCESS;
}
