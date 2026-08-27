/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file store_kv_block_new.h
 * \brief StoreKVBlockNew kernel operator
 */

#ifndef ASCEND_STORE_KV_BLOCK_NEW_H
#define ASCEND_STORE_KV_BLOCK_NEW_H

#include "kernel_operator.h"

namespace StoreKVBlockNew {
using namespace AscendC;


#ifndef STORE_KV_BLOCK_NEW_TILING_DATA_H_
#define STORE_KV_BLOCK_NEW_TILING_DATA_H_
struct StoreKVBlockNewTilingData{
    uint32_t blockTableSize;    
    uint32_t typeByte;
    uint32_t tokenSize;
    uint32_t numTokens;
    uint32_t numCache;
    uint32_t groupInfoLen;
    uint32_t maxTokensPerCopy;
};
#endif
template <typename T>
class StoreKVBlockNewBase {
public:

    uint32_t tokenSize = 0;
    uint32_t tokenByteSize = 0;
    uint32_t blockTableSize = 0;
    uint32_t typeByte = 0;
    uint32_t numTokens = 0;
    uint32_t numCache = 0;
    uint32_t groupInfoLen = 0;
    uint32_t maxTokensPerCopy = 0;

    uint32_t coreId = 0;
    uint32_t blockNum = 0;
    AscendC::TPipe* pipeThis;
    AscendC::LocalTensor<T> tokenLocal;
    AscendC::GlobalTensor<T> keyInputGt;
    AscendC::GlobalTensor<T> keyCacheInputGt;
    AscendC::GlobalTensor<int32_t> groupLenGt;
    AscendC::GlobalTensor<int32_t> groupKeyIdxGt;
    AscendC::GlobalTensor<int32_t> groupKeyCacheIdxGt;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf;
    __aicore__ inline StoreKVBlockNewBase() {}

    __aicore__ inline uint32_t RoundUp(uint32_t x, uint32_t y = 16)
    {
        return y == 0 ? 0 : (x + y - 1) / y * y;
    }

    __aicore__ inline void Init( AscendC::TPipe *pipe, StoreKVBlockNewTilingData *tilingData)
    {
        pipeThis = pipe;
        typeByte = tilingData->typeByte;
        tokenSize = tilingData->tokenSize;
        tokenByteSize = tokenSize*typeByte;
        blockTableSize = tilingData->blockTableSize;
        numTokens = tilingData->numTokens;
        numCache = tilingData->numCache;
        groupInfoLen = tilingData->groupInfoLen;
        maxTokensPerCopy = tilingData->maxTokensPerCopy;

        coreId = AscendC::GetBlockIdx();
        blockNum = AscendC::GetBlockNum();
    }
    __aicore__ inline void Process(GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR groupLen, GM_ADDR groupKeyIdx, GM_ADDR groupKeyCacheIdx)
    {
        
        keyInputGt.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(keyIn));
        keyCacheInputGt.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(keyCacheIn));
        groupLenGt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupLen));
        groupKeyIdxGt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupKeyIdx));
        groupKeyCacheIdxGt.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(groupKeyCacheIdx));

        // The UB buffer holds a single transfer at a time. A group larger than
        // maxTokensPerCopy tokens cannot be batched into one DataCopyPad with
        // blockCount>1: that single instruction would write blockCount*blockLen
        // bytes contiguously into UB (the whole group, e.g. 120 tokens for
        // maxTokensPerCopy=30, len=128), which exceeds the usable UB. So the
        // group is split into ceil(len/maxTokensPerCopy) single-chunk
        // (blockCount=1) transfers, each reusing this buffer.
        pipeThis->InitBuffer(tokenBuf, maxTokensPerCopy * tokenByteSize);
        tokenLocal = tokenBuf.Get<T>();

        AscendC::DataCopyExtParams copyParams{1, 0,  0, 0, 0};
        AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        // Distribute groups round-robin across cores: core coreId handles the
        // groups at indices coreId + i*blockNum. Each core loops only its own
        // share of groups and jumps directly to them, skipping invalid entries
        // - no full scan over all groups.
        uint32_t corePerNum = (groupInfoLen + blockNum - 1) / blockNum;
        uint32_t idx = 0;
        int32_t len = 0;
        int32_t src = 0;
        int32_t dst = 0;
        for (uint32_t i = 0; i < corePerNum; i++) {
            idx = coreId + i * blockNum;
            if (idx >= groupInfoLen) {
                continue;
            }
            len = groupLenGt.GetValue(idx);
            src = groupKeyIdxGt.GetValue(idx);
            dst = groupKeyCacheIdxGt.GetValue(idx);
            if (len <= 0 || src < 0 || dst < 0) {
                continue;
            }
            // Copy the group in maxTokensPerCopy-sized chunks, one DataCopyPad
            // (blockCount=1) per chunk. e.g. maxTokensPerCopy=30, len=128 ->
            // 5 transfers: 30+30+30+30+8. len/src/dst are non-negative after
            // validation; tokenSize/tokenByteSize/maxTokensPerCopy are uint32_t,
            // so src*tokenSize and the len>maxTokensPerCopy compare are done in
            // unsigned arithmetic via implicit promotion - no explicit type
            // conversion or extra variables needed.
            copyParams.blockCount = 1;
            while (len > 0) {
                int32_t chunk = (len > maxTokensPerCopy) ? maxTokensPerCopy : len;
                copyParams.blockLen = chunk * tokenByteSize;
                DataCopyPad(tokenLocal, keyInputGt[src * tokenSize], copyParams, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(EVENT_ID1);
                DataCopyPad(keyCacheInputGt[dst * tokenSize], tokenLocal, copyParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
                src += chunk;
                dst += chunk;
                len -= chunk;
            }
        }

    }

};
}

#endif
