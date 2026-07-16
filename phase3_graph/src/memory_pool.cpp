#include "memory_pool.h"
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <cstring>

#ifdef _WIN32
#include <malloc.h>  // _aligned_malloc
#endif

// ===========================================================================
// MemoryPool implementation
// ===========================================================================

MemoryPool::MemoryPool(size_t total_floats)
    : total_size_(total_floats), peak_used_(0), alloc_count_(0) {
    // 32-byte alignment for AVX2 loads (256-bit registers)
#ifdef _WIN32
    pool_ = static_cast<float*>(_aligned_malloc(total_floats * sizeof(float), 32));
#else
    pool_ = static_cast<float*>(std::aligned_alloc(32, total_floats * sizeof(float)));
#endif
    if (!pool_) {
        fprintf(stderr, "[MemoryPool] Failed to allocate %zu floats\n", total_floats);
    }
    // Zero-initialize so tensors start clean
    std::memset(pool_, 0, total_floats * sizeof(float));
}

MemoryPool::~MemoryPool() {
#ifdef _WIN32
    _aligned_free(pool_);
#else
    std::free(pool_);
#endif
}

// ===========================================================================
// Allocate: find a free slot, reuse if possible, otherwise append
// ===========================================================================

float* MemoryPool::allocate(const std::string& name, size_t count) {
    if (count > total_size_) {
        fprintf(stderr, "[MemoryPool] Request %zu > pool size %zu\n",
                count, total_size_);
        return nullptr;
    }

    // Strategy 1: Try to reuse a freed slot (best-fit)
    //   Walk through all slots, find a freed one that's big enough.
    //   This is the "reuse" part of memory pooling.
    for (auto& slot : slots_) {
        if (!slot.in_use && slot.size >= count) {
            slot.name = name;
            slot.in_use = true;
            alloc_count_++;
            return slot.ptr;
        }
    }

    // Strategy 2: Allocate a new slot at the end of the pool
    //   Calculate the end offset of all current slots.
    size_t offset = 0;
    for (const auto& slot : slots_) {
        size_t slot_end = (slot.ptr - pool_) + slot.size;
        offset = std::max(offset, slot_end);
    }

    if (offset + count > total_size_) {
        // Try compacting: consolidate all in-use slots to the front,
        // then try again.
        size_t compact_offset = 0;
        for (auto& slot : slots_) {
            if (slot.in_use) {
                if (slot.ptr != pool_ + compact_offset) {
                    std::memmove(pool_ + compact_offset, slot.ptr,
                                 slot.size * sizeof(float));
                    slot.ptr = pool_ + compact_offset;
                }
                compact_offset += slot.size;
            }
        }
        offset = compact_offset;

        if (offset + count > total_size_) {
            fprintf(stderr, "[MemoryPool] Out of memory after compaction: "
                    "%zu + %zu > %zu\n", offset, count, total_size_);
            return nullptr;
        }
    }

    Slot slot;
    slot.name = name;
    slot.ptr = pool_ + offset;
    slot.size = count;
    slot.in_use = true;

    slots_.push_back(slot);
    alloc_count_++;

    // Update peak usage
    size_t current = used_size();
    peak_used_ = std::max(peak_used_, current);

    return slot.ptr;
}

// ===========================================================================
// Deallocate: mark a slot as free (memory is NOT returned to OS)
// ===========================================================================

void MemoryPool::deallocate(const std::string& name) {
    for (auto& slot : slots_) {
        if (slot.in_use && slot.name == name) {
            slot.in_use = false;
            return;
        }
    }
}

// ===========================================================================
// Stats
// ===========================================================================

size_t MemoryPool::used_size() const {
    size_t total = 0;
    for (const auto& slot : slots_) {
        if (slot.in_use) total += slot.size;
    }
    return total;
}

void MemoryPool::dump() const {
    printf("=== MemoryPool: %zu/%zu floats used (peak=%zu, %zu allocs) ===\n",
           used_size(), total_size_, peak_used_, alloc_count_);
    for (const auto& slot : slots_) {
        printf("  [%s] %s offset=%zu size=%zu\n",
               slot.in_use ? "IN USE" : "FREE ",
               slot.name.c_str(),
               (size_t)(slot.ptr - pool_),
               slot.size);
    }
}
