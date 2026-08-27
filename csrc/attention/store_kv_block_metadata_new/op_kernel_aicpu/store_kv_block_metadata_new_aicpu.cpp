/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You should not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file store_kv_block_metadata_new_aicpu.cpp
 * \brief AICPU kernel implementation for StoreKvBlockMetadataNew
 *
 * Ports the logic from store_kv_block_pre: groups contiguous slot_mapping entries
 * that belong to the same block, producing group_len / group_key_idx / group_key_cache_idx.
 */

#include "log.h"
#include "status.h"
#include "cpu_kernel_utils.h"
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include "store_kv_block_metadata_new_aicpu.h"

namespace aicpu {

// Below this input length the per-thread dispatch overhead of ParallelFor is not
// amortized, so the dedup runs on a single thread.
constexpr int64_t kParallelDataNum = 8 * 1024;

// Grouping is only a per-element comparison (unlike hash-based dedup), so a serial
// scan beats ParallelFor until the input is much larger than kParallelDataNum: the
// per-chunk dispatch cost dwarfs the actual scan work. Below this the single-threaded
// GroupConsecutiveSlots runs.
constexpr int64_t kParallelGroupDataNum = 80 * 1024;

// Dedup pre-scan: resolve duplicate slot_mapping entries so the LAST occurrence wins.
// e.g. key[1] and key[100] may both target cache[66]; we want key[100] (the later
// one) to overwrite. Every earlier occurrence of a slot is marked -1 so the grouping
// loop below skips it.
static void DedupSlotMapping(int32_t *slotMappingData, int64_t slotMappingLen)
{
    // Fast path: slot_mapping grows monotonically as tokens are appended to cache
    // blocks, so in practice its positive values are already sorted (non-decreasing)
    // and duplicates are adjacent and rare. For such input dedup needs no hash table:
    // a single linear pass marks the previous occurrence whenever the current entry
    // equals it. Dirty (-1) entries are skipped WITHOUT resetting the running slot, so
    // a -1 gap between two equal slots still demotes the earlier copy. Demoting the
    // previous index is always safe - it can never be the LAST occurrence of a value -
    // so the same pass doubles as a sortedness probe: if a positive value ever drops,
    // fall back to the general hash pass below to stay correct on unsorted input.
    int32_t prevSlot = -1;
    int64_t prevIdx = -1;
    bool sorted = true;
    for (int64_t i = 0; i < slotMappingLen; ++i) {
        const int32_t slot = slotMappingData[i];
        if (slot < 0) {
            continue;
        }
        if (slot < prevSlot) {
            sorted = false;
        } else if (slot == prevSlot) {
            // Adjacent (in the positive subsequence) duplicate: the earlier copy is
            // not the last occurrence of this slot, so demote it right here.
            slotMappingData[prevIdx] = -1;
        }
        prevSlot = slot;
        prevIdx = i;
    }
    if (sorted) {
        return;
    }

    // General path (unsorted input): open-addressing hash table built over two
    // contiguous arrays, sized >= 2*slotMappingLen so the load factor stays <= 0.5
    // even when every slot is distinct (probes stay 1-2 steps). Replaces
    // std::unordered_map: no per-node heap allocation, no pointer-chasing chains, so
    // a fully unsorted input stays cheap instead of paying a malloc per distinct slot.
    size_t tableSize = 1;
    while (tableSize < static_cast<size_t>(2 * slotMappingLen)) {
        tableSize <<= 1;
    }
    std::vector<int32_t> tableSlot(tableSize, -1);  // -1 = empty; valid slots are >= 0
    std::vector<int32_t> tableIdx(tableSize, -1);
    const size_t tableMask = tableSize - 1;
    auto slotHash = [](int32_t s) -> size_t {
        return static_cast<size_t>(s) * 0x9E3779B1u;  // Knuth multiplicative hash
    };

    // Insert pass: record the LAST index of every distinct slot.
    for (int64_t i = 0; i < slotMappingLen; ++i) {
        const int32_t slot = slotMappingData[i];
        if (slot < 0) {
            continue;
        }
        size_t p = slotHash(slot) & tableMask;
        for (;;) {
            if (tableSlot[p] == -1) {
                tableSlot[p] = slot;
                tableIdx[p] = static_cast<int32_t>(i);
                break;
            }
            if (tableSlot[p] == slot) {
                if (i > tableIdx[p]) {
                    tableIdx[p] = static_cast<int32_t>(i);
                }
                break;
            }
            p = (p + 1) & tableMask;
        }
    }
    // Mark pass: any entry whose index is not the recorded last occurrence is a
    // duplicate and gets demoted to -1.
    for (int64_t i = 0; i < slotMappingLen; ++i) {
        const int32_t slot = slotMappingData[i];
        if (slot < 0) {
            continue;
        }
        size_t p = slotHash(slot) & tableMask;
        for (;;) {
            if (tableSlot[p] == -1) {
                break;  // not found: cannot happen after the insert pass
            }
            if (tableSlot[p] == slot) {
                if (tableIdx[p] != i) {
                    slotMappingData[i] = -1;
                }
                break;
            }
            p = (p + 1) & tableMask;
        }
    }
}

// Two-level hash alternative to DedupSlotMapping. A small cache-resident level-0
// table (power-of-two, kept below half full) absorbs the hot slots with a cheap
// multiplicative hash; everything it cannot take (new keys once level 0 is at
// capacity) goes into a larger level-1 table (sized >= 2*slotMappingLen, so load
// stays <= 0.5 even when every slot is distinct). The two levels use different
// multiplicative hashes, and each key lives in exactly one level, so both insert
// and lookup are short open-address probes over contiguous arrays: no per-node
// allocation, no collision chains, cache-friendly.
[[maybe_unused]] static void DedupSlotMappingTwoLevel(int32_t *slotMappingData, int64_t slotMappingLen)
{
    constexpr size_t kL0Size = 1024;  // level-0 size, power of two
    std::vector<int32_t> l0Slot(kL0Size, -1);  // -1 = empty; valid slots are >= 0
    std::vector<int32_t> l0Idx(kL0Size, -1);
    size_t l0Count = 0;

    // Level-1 capacity: next power of two >= 2*slotMappingLen.
    size_t l1Size = 1;
    while (l1Size < static_cast<size_t>(2 * slotMappingLen)) {
        l1Size <<= 1;
    }
    std::vector<int32_t> l1Slot(l1Size, -1);
    std::vector<int32_t> l1Idx(l1Size, -1);

    const size_t l0Mask = kL0Size - 1;
    const size_t l1Mask = l1Size - 1;

    auto hash0 = [](int32_t s) -> size_t {
        return static_cast<size_t>(s) * 0x9E3779B1u;
    };
    auto hash1 = [](int32_t s) -> size_t {
        uint32_t x = static_cast<uint32_t>(s);
        x ^= x >> 16;
        return static_cast<size_t>(x) * 0x85EBCA6Bu;
    };

    auto insert = [&](int32_t slot, int32_t idx) {
        // Refresh path: if the key is already in level 0, update it there so each
        // key keeps a single copy in one level.
        size_t p0 = hash0(slot) & l0Mask;
        for (;;) {
            if (l0Slot[p0] == -1) {
                break;
            }
            if (l0Slot[p0] == slot) {
                if (idx > l0Idx[p0]) {
                    l0Idx[p0] = idx;
                }
                return;
            }
            p0 = (p0 + 1) & l0Mask;
        }
        // New key: level 0 while it has room, otherwise level 1.
        if (l0Count < kL0Size / 2) {
            l0Slot[p0] = slot;
            l0Idx[p0] = idx;
            ++l0Count;
            return;
        }
        size_t p1 = hash1(slot) & l1Mask;
        for (;;) {
            if (l1Slot[p1] == -1) {
                l1Slot[p1] = slot;
                l1Idx[p1] = idx;
                return;
            }
            if (l1Slot[p1] == slot) {
                if (idx > l1Idx[p1]) {
                    l1Idx[p1] = idx;
                }
                return;
            }
            p1 = (p1 + 1) & l1Mask;
        }
    };

    auto findIdx = [&](int32_t slot) -> int32_t {
        size_t p0 = hash0(slot) & l0Mask;
        for (;;) {
            if (l0Slot[p0] == -1) {
                break;
            }
            if (l0Slot[p0] == slot) {
                return l0Idx[p0];
            }
            p0 = (p0 + 1) & l0Mask;
        }
        size_t p1 = hash1(slot) & l1Mask;
        for (;;) {
            if (l1Slot[p1] == -1) {
                return -1;
            }
            if (l1Slot[p1] == slot) {
                return l1Idx[p1];
            }
            p1 = (p1 + 1) & l1Mask;
        }
    };

    for (int64_t i = 0; i < slotMappingLen; ++i) {
        const int32_t slot = slotMappingData[i];
        if (slot >= 0) {
            insert(slot, static_cast<int32_t>(i));
        }
    }
    for (int64_t i = 0; i < slotMappingLen; ++i) {
        const int32_t slot = slotMappingData[i];
        if (slot >= 0 && findIdx(slot) != i) {
            slotMappingData[i] = -1;
        }
    }
}

// Parallel hash alternative to DedupSlotMapping: same unordered_map semantics but
// parallelized with CpuKernelUtils::ParallelFor. ParallelFor slices the range into
// an arbitrary number of shards; each shard demotes its own duplicates inline (same
// thread, safe) while building a private (slot -> max index) table, then merges it
// into the shared table under a single lock per shard, recording any superseded
// global max in a demote list instead of touching the data. A final parallel pass
// visits ONLY that list - i.e. only the real duplicates - and marks them -1, with no
// full second scan over the input. Falls back to the serial single-pass DedupSlotMapping
// if the input is too small or ParallelFor fails.
[[maybe_unused]] static void DedupSlotMappingParallel(int32_t *slotMappingData, int64_t slotMappingLen,
                                                     CpuKernelContext &ctx)
{
    if (slotMappingLen < kParallelDataNum) {
        DedupSlotMapping(slotMappingData, slotMappingLen);
        return;
    }

    std::unordered_map<int32_t, int32_t> lastIdx;
    std::vector<int32_t> demoteList;  // global indices that are no longer the last occurrence
    std::mutex mergeMtx;

    auto recordTask = [&](int64_t start, int64_t end) {
        std::unordered_map<int32_t, int32_t> local;
        for (int64_t i = start; i < end; ++i) {
            const int32_t slot = slotMappingData[i];
            if (slot < 0) {
                continue;
            }
            auto ret = local.emplace(slot, static_cast<int32_t>(i));
            if (!ret.second) {
                // Same slot reappears inside this shard: the previous local max is no
                // longer the last occurrence. Same thread + disjoint range, so demote it
                // right here without touching the shared state.
                slotMappingData[ret.first->second] = -1;
                ret.first->second = static_cast<int32_t>(i);
            }
        }
        std::lock_guard<std::mutex> lock(mergeMtx);
        for (const auto &kv : local) {
            auto git = lastIdx.find(kv.first);
            if (git == lastIdx.end()) {
                lastIdx.emplace(kv.first, kv.second);
            } else if (kv.second > git->second) {
                demoteList.push_back(git->second);  // superseded global max is a duplicate
                git->second = kv.second;
            }
        }
    };

    auto demoteTask = [&](int64_t start, int64_t end) {
        for (int64_t i = start; i < end; ++i) {
            slotMappingData[demoteList[i]] = -1;
        }
    };

    const int64_t maxCoreNum =
        std::min(slotMappingLen, static_cast<int64_t>(CpuKernelUtils::GetCPUNum(ctx)));
    const int64_t perUnitSize = (slotMappingLen + maxCoreNum - 1) / maxCoreNum;
    if (CpuKernelUtils::ParallelFor(ctx, slotMappingLen, perUnitSize, recordTask) != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor(record) failed, fall back to serial.");
        DedupSlotMapping(slotMappingData, slotMappingLen);
        return;
    }

    const int64_t demoteLen = static_cast<int64_t>(demoteList.size());
    if (demoteLen > 0) {
        const int64_t demoteCoreNum =
            std::min(demoteLen, static_cast<int64_t>(CpuKernelUtils::GetCPUNum(ctx)));
        const int64_t demotePerUnitSize = (demoteLen + demoteCoreNum - 1) / demoteCoreNum;
        if (CpuKernelUtils::ParallelFor(ctx, demoteLen, demotePerUnitSize, demoteTask) !=
            KERNEL_STATUS_OK) {
            KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor(demote) failed, fall back to serial.");
            for (int32_t d : demoteList) {
                slotMappingData[d] = -1;
            }
        }
    }
}

uint32_t StoreKvBlockMetadataNewCpuKernel::Compute(CpuKernelContext &ctx)
{
    bool success = Prepare(ctx);
    if (!success) {
        return KERNEL_STATUS_PARAM_INVALID;
    }
    return GenMetaData(ctx) ? KERNEL_STATUS_OK : KERNEL_STATUS_PARAM_INVALID;
}

bool StoreKvBlockMetadataNewCpuKernel::Prepare(CpuKernelContext &ctx)
{
    // inputs
    slotMapping_ = ctx.Input(static_cast<uint32_t>(ParamId::slotMapping));
    groupLen_ = ctx.Input(static_cast<uint32_t>(ParamId::groupLen));
    groupKeyIdx_ = ctx.Input(static_cast<uint32_t>(ParamId::groupKeyIdx));
    groupKeyCacheIdx_ = ctx.Input(static_cast<uint32_t>(ParamId::groupKeyCacheIdx));

    // attribute
    auto attr = ctx.GetAttr("block_size");
    if (attr == nullptr) {
        KERNEL_LOG_ERROR("attr block_size is null");
        return false;
    }
    blockSize_ = static_cast<int32_t>(attr->GetInt());
    if (blockSize_ <= 0) {
        KERNEL_LOG_ERROR("block_size must be positive, got %d", blockSize_);
        return false;
    }
    return true;
}

// Group consecutive slot_mapping entries that belong to the same cache block,
// producing groupLen / groupKeyIdx / groupKeyCacheIdx. Returns the number of
// groups written; the caller uses it to 0-fill the remaining output entries.
static int32_t GroupConsecutiveSlots(const int32_t *slotMappingData, int64_t slotMappingLen,
                                     int32_t blockSize, int32_t *groupLenData,
                                     int32_t *groupKeyIdxData, int32_t *groupKeyCacheIdxData)
{
    int32_t idxSlotmap = 0;
    int32_t idxGroups = 0;

    while (idxSlotmap < slotMappingLen) {
        // Skip dirty values (negative slots)
        int32_t cacheSlot = slotMappingData[idxSlotmap];
        if (cacheSlot < 0) {
            idxSlotmap++;
            continue;
        }

        int32_t blockId = cacheSlot / blockSize;

        // Record group start: source index and destination cache index
        groupKeyIdxData[idxGroups] = idxSlotmap;
        groupKeyCacheIdxData[idxGroups] = cacheSlot;

        // Find the end of consecutive slots within the same block
        int32_t groupEndIdx = idxSlotmap;
        while (groupEndIdx + 1 < slotMappingLen
               && slotMappingData[groupEndIdx + 1] / blockSize == blockId
               && slotMappingData[groupEndIdx + 1] == slotMappingData[groupEndIdx] + 1) {
            groupEndIdx++;
        }
        groupEndIdx++;

        groupLenData[idxGroups] = groupEndIdx - idxSlotmap;

        idxSlotmap = groupEndIdx;
        idxGroups++;
    }
    return idxGroups;
}

// Parallel variant of GroupConsecutiveSlots. A group is a maximal run of
// consecutive slots that also stay within the same cache block.
// A chunk boundary is a hard stop: a run that continues into the next chunk is
// emitted as a separate group (continuity across threads is intentionally ignored).
//   1. Each thread counts the group starts in its own chunk (O(chunk) with no
//      per-group allocations) into a tiny per-chunk counter.
//   2. A short serial pass turns those counts into a prefix sum = each chunk's
//      contiguous output base.
//   3. Each thread writes its groups directly into its own contiguous output range
//      [chunkBase[ci], chunkBase[ci]+count) - sequential writes, no shared state,
//      no local buffers and no merge pass.
// Falls back to the serial GroupConsecutiveSlots for small inputs or if ParallelFor fails.
static int32_t GroupConsecutiveSlotsParallel(const int32_t *slotMappingData, int64_t slotMappingLen,
                                             int32_t blockSize, int32_t *groupLenData,
                                             int32_t *groupKeyIdxData, int32_t *groupKeyCacheIdxData,
                                             CpuKernelContext &ctx)
{
    if (slotMappingLen < kParallelGroupDataNum) {
        return GroupConsecutiveSlots(slotMappingData, slotMappingLen, blockSize, groupLenData,
                                     groupKeyIdxData, groupKeyCacheIdxData);
    }

    const int64_t maxCoreNum =
        std::min(slotMappingLen, static_cast<int64_t>(CpuKernelUtils::GetCPUNum(ctx)));
    const int64_t perUnitSize = (slotMappingLen + maxCoreNum - 1) / maxCoreNum;
    const int64_t numChunks = (slotMappingLen + perUnitSize - 1) / perUnitSize;

    // Pass 1: per-chunk group counts, computed in parallel.
    std::vector<int64_t> chunkCount(numChunks, 0);
    auto countTask = [&](int64_t start, int64_t end) {
        const int64_t ci = start / perUnitSize;
        int64_t count = 0;
        for (int64_t i = start; i < end; ++i) {
            if (slotMappingData[i] < 0) {
                continue;
            }
            // A group starts at a chunk boundary (cross-chunk continuity is
            // ignored), after a dirty slot, or when the previous slot is neither
            // consecutive nor in the same block.
            if (i == start || slotMappingData[i - 1] < 0 ||
                slotMappingData[i] != slotMappingData[i - 1] + 1 ||
                slotMappingData[i] / blockSize != slotMappingData[i - 1] / blockSize) {
                ++count;
            }
        }
        chunkCount[ci] = count;
    };

    if (CpuKernelUtils::ParallelFor(ctx, slotMappingLen, perUnitSize, countTask) != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor(count) failed, fall back to serial.");
        return GroupConsecutiveSlots(slotMappingData, slotMappingLen, blockSize, groupLenData,
                                     groupKeyIdxData, groupKeyCacheIdxData);
    }

    // Prefix sum: chunk group counts -> global output base per chunk.
    std::vector<int64_t> chunkBase(numChunks + 1, 0);
    for (int64_t k = 0; k < numChunks; ++k) {
        chunkBase[k + 1] = chunkBase[k] + chunkCount[k];
    }

    // Pass 2: each thread writes its groups into its own contiguous output range.
    auto writeTask = [&](int64_t start, int64_t end) {
        const int64_t ci = start / perUnitSize;
        int64_t g = chunkBase[ci];
        int64_t i = start;
        while (i < end) {
            const int32_t cur = slotMappingData[i];
            if (cur < 0) {
                ++i;
                continue;
            }
            const int64_t groupStart = i;
            // Extend only over consecutive slots within the same block.
            while (i + 1 < end && slotMappingData[i + 1] == slotMappingData[i] + 1 &&
                   slotMappingData[i + 1] / blockSize == slotMappingData[i] / blockSize) {
                ++i;
            }
            ++i;
            groupKeyIdxData[g] = static_cast<int32_t>(groupStart);
            groupKeyCacheIdxData[g] = slotMappingData[groupStart];
            groupLenData[g] = static_cast<int32_t>(i - groupStart);
            ++g;
        }
    };

    if (CpuKernelUtils::ParallelFor(ctx, slotMappingLen, perUnitSize, writeTask) != KERNEL_STATUS_OK) {
        KERNEL_LOG_ERROR("CpuKernelUtils::ParallelFor(write) failed, fall back to serial.");
        return GroupConsecutiveSlots(slotMappingData, slotMappingLen, blockSize, groupLenData,
                                     groupKeyIdxData, groupKeyCacheIdxData);
    }
    return static_cast<int32_t>(chunkBase[numChunks]);
}

bool StoreKvBlockMetadataNewCpuKernel::GenMetaData(CpuKernelContext &ctx)
{
    if (slotMapping_ == nullptr || slotMapping_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("slot_mapping is empty");
        return false;
    }
    if (groupLen_ == nullptr || groupLen_->GetData() == nullptr ||
        groupKeyIdx_ == nullptr || groupKeyIdx_->GetData() == nullptr ||
        groupKeyCacheIdx_ == nullptr || groupKeyCacheIdx_->GetData() == nullptr) {
        KERNEL_LOG_ERROR("input tensor is empty");
        return false;
    }

    int32_t *slotMappingData = static_cast<int32_t *>(slotMapping_->GetData());
    int32_t *groupLenData = static_cast<int32_t *>(groupLen_->GetData());
    int32_t *groupKeyIdxData = static_cast<int32_t *>(groupKeyIdx_->GetData());
    int32_t *groupKeyCacheIdxData = static_cast<int32_t *>(groupKeyCacheIdx_->GetData());

    // total elements in slot_mapping (1-D tensor)
    int64_t slotMappingLen = slotMapping_->GetTensorShape()->GetDimSize(0);

    // total capacity of output tensors (1-D, same shape as input)
    int64_t outCapacity = groupLen_->GetTensorShape()->GetDimSize(0);

    // Dedup pre-scan so duplicate slots (last occurrence wins) don't confuse grouping.
    // DedupSlotMapping(slotMappingData, slotMappingLen);

    int32_t idxGroups = GroupConsecutiveSlotsParallel(slotMappingData, slotMappingLen, blockSize_,
                                                      groupLenData, groupKeyIdxData, groupKeyCacheIdxData,
                                                      ctx);

    // 0 fill the remaining output entries. store_kv_block kernel reads groupLen as uint32_t,
    // so negative fillers would be interpreted as huge positive values and bypass the
    // `groupLen <= 0` guard, causing out-of-range MTE writes. Use 0 so that guard works.
    if (idxGroups < outCapacity) {
        std::memset(groupLenData + idxGroups, 0,
                    static_cast<size_t>(outCapacity - idxGroups) * sizeof(int32_t));
        std::memset(groupKeyIdxData + idxGroups, 0,
                    static_cast<size_t>(outCapacity - idxGroups) * sizeof(int32_t));
        std::memset(groupKeyCacheIdxData + idxGroups, 0,
                    static_cast<size_t>(outCapacity - idxGroups) * sizeof(int32_t));
    }

    return true;
}

namespace {
static const char *kernelType = "StoreKvBlockMetadataNew";
REGISTER_CPU_KERNEL(kernelType, StoreKvBlockMetadataNewCpuKernel);
} // namespace

} // namespace aicpu
