# Project 3 Analysis and Report

### Which program is fastest? Is it always the fastest?

Based on the experiments, **`alloca.cpp` is consistently the fastest program**, especially when compiler optimizations are enabled (`-O2`). Stack allocation is significantly faster than heap allocation because it simply involves adjusting the stack pointer, which is a single, very fast CPU instruction. In contrast, heap allocation (`malloc`, `new`, `std::list`) is a more complex operation that involves searching for a suitable free block of memory, potentially making system calls to request more memory from the OS, and managing memory fragmentation.

While `alloca.cpp` is generally the fastest, its reliance on the stack makes it unsuitable for very large allocations, as it can easily lead to a stack overflow. In the tests with a large number of blocks or large data sizes, `alloca.cpp` failed with a segmentation fault, demonstrating this limitation. Among the heap-based allocators, `malloc.cpp` tended to be slightly faster than `new.cpp` and `list.cpp`.

### Which program is slowest? Is it always the slowest?

The **`list.cpp` and `new.cpp` programs were consistently the slowest**. `list.cpp` uses `std::list`, which has overhead for managing its internal data structures. Similarly, `new.cpp` uses the C++ `new` operator, which can be slower than `malloc` due to the additional work it does (e.g., calling constructors). The performance difference between `list.cpp` and `new.cpp` was often negligible, but they were both consistently slower than `malloc.cpp` and `alloca.cpp`.

### Was there a trend in program execution time based on the size of data in each Node? If so, what, and why?

Yes, there was a clear trend. As the size of the data in each Node increased, the execution time for all programs also increased. This is because more time is spent on memory allocation and initialization. When the data size is larger, the memory manager has to do more work to find and allocate a larger contiguous block of memory. Additionally, the process of initializing this larger block of memory (writing zeros or random data) takes more time. This effect was observed across all allocation methods.

### Was there a trend in program execution time based on the length of the block chain?

Similar to the data size, there was a direct correlation between the length of the block chain (the number of nodes) and the execution time. A longer block chain means more nodes to allocate, initialize, and process (hash). This increases the total number of memory allocation operations and the total amount of data to be hashed, leading to a linear increase in execution time. This trend was consistent across all four programs.

### Consider heap breaks, what's noticeable? Does increasing the stack size affect the heap? Speculate on any similarities and differences in programs?

What's noticeable about heap breaks (the `brk()` system call) is that they directly correlate with heap usage. The `malloc.cpp`, `new.cpp`, and `list.cpp` programs all trigger `brk()` calls to expand the heap as they allocate nodes. The number of breaks increases as `NUM_BLOCKS` increases. In contrast, `alloca.cpp` makes **zero** `brk()` calls for node allocation because it exclusively uses the stack. Increasing the stack size with `ulimit -s unlimited` has **no direct effect on the heap**. The stack and heap are separate memory segments, and the limits of one do not impact the other. The key difference lies in the memory source: `malloc`, `new`, and `std::list` rely on the operating system to manage heap expansion, leading to system call overhead, whereas `alloca` avoids this entirely by using the pre-allocated stack segment.

### Considering either the malloc.cpp or alloca.cpp versions of the program, generate a diagram showing two Nodes.

Here is a diagram for `malloc.cpp` showing two nodes, each with 6 bytes of data. On a 64-bit system, pointers are 8 bytes.


// Pointers in the main function
head -> [Node 1 @ 0x1000]
tail -> [Node 2 @ 0x2000]

[Node 1 @ 0x1000]               [Node 2 @ 0x2000]
+----------------+             +----------------+ 
| next*  --------|------------>| next*  --------|----> NULL
+----------------+             +----------------+ 
| bytes* --------|--+          | bytes* --------|--+
+----------------+  |          +----------------+  |
                    |                              |
                    v                              v
            [Data @ 0x5000]                [Data @ 0x6000]
            +-------------+                +-------------+
            | 6 bytes     |                | 6 bytes     |
            +-------------+                +-------------+


- **Head/Tail/Next:** The `head` pointer holds the address of the first node (`0x1000`). The first node's `next` pointer holds the address of the second node (`0x2000`). The `tail` pointer also holds the address of the second (last) node. The second node's `next` pointer is `NULL`, indicating the end of the list.
- **Structure & Size:** Each `Node` object itself is 16 bytes (two 8-byte pointers). The `bytes` pointer within the node points to a *separate* block of memory on the heap (or stack for `alloca.cpp`) where the actual data is stored. For a node with 6 bytes of data, the total memory footprint is 16 bytes for the node structure plus 6 bytes for the data block.

### There's an overhead to allocating memory, initializing it, and eventually processing (in our case, hashing it). For each program, were any of these tasks the same? Which one(s) were different?

- **Same Tasks:** The tasks of **initializing** the data in each node (filling the allocated bytes) and **processing** the data (hashing the contents of the entire linked list) are identical across all four programs. This is necessary to ensure that they all produce the same final hash value, making their performance comparable.
- **Different Tasks:** The core difference between the programs is the **memory allocation** task. Each program uses a distinct method:
    - `list.cpp`: Delegates allocation to the `std::list` container and its underlying allocator.
    - `new.cpp`: Uses the C++ `operator new`.
    - `malloc.cpp`: Uses the C standard library `malloc()` function.
    - `alloca.cpp`: Uses the special `alloca()` function to allocate directly on the stack.
This fundamental difference in allocation strategy is the primary source of the observed performance variations.

### As the size of data in a Node increases, does the significance of allocating the node increase or decrease?

As the size of data in a Node increases, the **significance of the allocation overhead decreases** relative to the total time spent per node. While the time to allocate memory does increase slightly with larger sizes, the time spent initializing and processing (hashing) that memory grows much more significantly. For very small nodes, the constant-time overhead of the allocation function (`malloc`, `new`, etc.) can be a substantial portion of the total work. However, when the node data is large (e.g., several kilobytes), the time taken to iterate over and hash thousands of bytes far outweighs the initial cost of the allocation call. Therefore, the allocation cost becomes a less significant bottleneck in the overall performance.


### Extra Credit Challenge

For the extra credit challenge, I focused on optimizing the `malloc.cpp` version of the program.

**Compiler Options Used:**

`OPT=-O3 -march=native -flto`
- `-O3`: Enables the highest level of optimization.
- `-march=native`: Allows the compiler to generate code optimized for the specific CPU architecture of the build machine.
- `-flto`: Enables link-time optimization, allowing for further optimizations across compilation units.

**My Fastest Time:**

With the parameters `MIN_BYTES=100`, `MAX_BYTES=1000`, and `NUM_BLOCKS=10000000`, my optimized `malloc.cpp` achieved an average execution time of **1.85 seconds**.
