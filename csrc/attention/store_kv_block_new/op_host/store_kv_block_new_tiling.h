#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(StoreKVBlockNewTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, blockTableSize);
    TILING_DATA_FIELD_DEF(uint32_t, typeByte);
    TILING_DATA_FIELD_DEF(uint32_t, tokenSize);
    TILING_DATA_FIELD_DEF(uint32_t, numTokens);
    TILING_DATA_FIELD_DEF(uint32_t, numCache);
    TILING_DATA_FIELD_DEF(uint32_t, groupInfoLen);
    TILING_DATA_FIELD_DEF(uint32_t, maxTokensPerCopy);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(StoreKVBlockNew, StoreKVBlockNewTilingData)

struct StoreKVBlockNewCompileInfo {
    uint32_t coreNum;
    uint64_t ubSizePlatForm;
    uint32_t sysWorkspaceSize;
};

} // namespace optiling
