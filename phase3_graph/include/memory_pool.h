#ifndef PHASE3_MEMORY_POOL_H
#define PHASE3_MEMORY_POOL_H

#include <vector>
#include <string>
#include <cstdio>
#include <cstring>

// ===========================================================================
// Memory Pool for intermediate tensor allocation
// ===========================================================================
//
// Problem:
//   Each operator in the graph allocates a new output tensor via
//   std::vector<float>. This means:
//     1. malloc/free overhead for every intermediate tensor
//     2. Memory fragmentation from frequent alloc/dealloc
//     3. Peak memory = sum of all intermediate tensors (no reuse)
//
// Solution:
//   Pre-allocate one big buffer. Each tensor "borrows" a slice from the
//   buffer. After a tensor's last consumer runs, its slice can be reclaimed
//   and reused by later tensors.
//
//   This is the same technique used by TensorRT, ONNX Runtime, and
//   llama.cpp for inference memory management.
//
// Architecture:
//   ┌───────────────────── Big Buffer ─────────────────────┐
//   │ [tensor_A] [tensor_B] [tensor_C] [.....free.....]   │
//   │     ↑           ↑           ↑                        │
//   │   in use     in use      in use                      │
//   │                                                      │
//   │ When B is no longer needed, its slot is freed:       │
//   │ [tensor_A] [..free..] [tensor_C] [.....free.....]   │
//   │ Then tensor_D can reuse B's old slot:                │
//   │ [tensor_A] [tensor_D] [tensor_C] [.....free.....]   │
//   └──────────────────────────────────────────────────────┘
//
// Key concept: Lifetime analysis
//   A tensor's lifetime = [first_use_node_index, last_use_node_index]
//   Two tensors with non-overlapping lifetimes can share the same memory.
// ===========================================================================

class MemoryPool {
public:
    // Allocate a pool of given total size (in number of floats)
    explicit MemoryPool(size_t total_floats);
    ~MemoryPool();

    // Allocate a slice of the pool. Returns nullptr if out of memory.
    // The caller provides a name for debugging and the number of floats needed.
    float* allocate(const std::string& name, size_t count);

    // Free a slice (mark it as reusable)
    void deallocate(const std::string& name);

    // Get the base pointer of the pool (for direct access)
    float* base() { return pool_; }

    // Stats
    size_t total_size() const { return total_size_; }
    size_t used_size() const;
    size_t free_size() const { return total_size_ - used_size(); }
    size_t peak_used() const { return peak_used_; }
    size_t allocation_count() const { return alloc_count_; }

    void dump() const;

private:
    struct Slot {
        std::string name;
        float* ptr;
        size_t size;   // in number of floats
        bool in_use;
    };

    float* pool_;              // the big buffer
    size_t total_size_;        // total floats in pool
    size_t peak_used_;         // max floats ever in use at once
    size_t alloc_count_;       // total number of allocations made
    std::vector<Slot> slots_;  // all allocated slots
};

#endif // PHASE3_MEMORY_POOL_H
