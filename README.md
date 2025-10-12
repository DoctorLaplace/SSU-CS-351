# Project 1 Analysis Report

### Which program is fastest? Is it always the fastest?

Based on the experiments, **`alloca.cpp` is consistently the fastest program**, especially when compiler optimizations are enabled (`-O2`). Stack allocation is significantly faster than heap allocation because it simply involves adjusting the stack pointer, which is a single, very fast CPU instruction. In contrast, heap allocation (`malloc`, `new`, `std::list`) is a more complex operation that involves searching for a suitable free block of memory, potentially making system calls to request more memory from the OS, and managing memory fragmentation.

While `alloca.cpp` is generally the fastest, its reliance on the stack makes it unsuitable for very large allocations, as it can easily lead to a stack overflow. In the tests with a large number of blocks or large data sizes, `alloca.cpp` failed with a segmentation fault, demonstrating this limitation. Among the heap-based allocators, `malloc.cpp` tended to be slightly faster than `new.cpp` and `list.cpp`.

### Which program is slowest? Is it always the slowest?

The **`list.cpp` and `new.cpp` programs were consistently the slowest**. `list.cpp` uses `std::list`, which has overhead for managing its internal data structures. Similarly, `new.cpp` uses the C++ `new` operator, which can be slower than `malloc` due to the additional work it does (e.g., calling constructors). The performance difference between `list.cpp` and `new.cpp` was often negligible, but they were both consistently slower than `malloc.cpp` and `alloca.cpp`.

### Was there a trend in program execution time based on the size of data in each Node? If so, what, and why?

Yes, there was a clear trend. As the size of the data in each Node increased, the execution time for all programs also increased. This is because more time is spent on memory allocation and initialization. When the data size is larger, the memory manager has to do more work to find and allocate a larger contiguous block of memory. Additionally, the process of initializing this larger block of memory (writing zeros or random data) takes more time. This effect was observed across all allocation methods.

### Was there a trend in program execution time based on the length of the block chain?

Similar to the data size, there was a direct correlation between the length of the block chain (the number of nodes) and the execution time. A longer block chain means more nodes to allocate, initialize, and process (hash). This increases the total number of memory allocation operations and the total amount of data to be hashed, leading to a linear increase in execution time. This trend was consistent across all four programs.

### Node Diagram (for malloc.cpp/alloca.cpp with 6 bytes of data)

Below is a textual representation of the memory layout for two connected `Node` objects.


head ----> [Node 1]
             |
             +---------------- next* --------------+     tail ---+ 
             |                                       |             |
             |       [Node 2] <--------------------+             |
             |          |                                            |
             |          +---------------- next* ----> NULL               |
             |          |                                            |
             |          +---------------- bytes* ----> [ 6 bytes data ]    |
             |                                                       |
             +---------------- bytes* ----> [ 6 bytes data ] <----------+

### Structure of a Single Node (allocating 6 bytes)

A single `Node` object itself contains two pointers. On a 64-bit system, each pointer is 8 bytes.

Memory Address | Field      | Size    | Points To
----------------------------------------------------------------
0x1000         | next       | 8 bytes | Address of the next Node (e.g., 0x1010)
0x1008         | bytes      | 8 bytes | Address of the allocated data block (e.g., 0x2000)

The `bytes` pointer points to a separate block of memory on the heap (for `malloc`) or stack (for `alloca`).

Memory Address | Content
--------------------------
0x2000         | data byte 0
0x2001         | data byte 1
0x2002         | data byte 2
0x2003         | data byte 3
0x2004         | data byte 4
0x2005         | data byte 5


### As the size of data in a Node increases, does the significance of allocating the node increase or decrease?

As the size of data in a Node increases, the **significance of the allocation overhead decreases** relative to the total time spent per node. While the time to allocate memory does increase slightly with larger sizes, the time spent initializing and processing (hashing) that memory grows much more significantly. For very small nodes, the constant-time overhead of the allocation function (`malloc`, `new`, etc.) can be a substantial portion of the total work. However, when the node data is large (e.g., several kilobytes), the time taken to iterate over and hash thousands of bytes far outweighs the initial cost of the allocation call. Therefore, the allocation cost becomes a less significant bottleneck in the overall performance.

