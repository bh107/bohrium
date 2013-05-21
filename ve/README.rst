===================================
Brief Description of Vector Engines
===================================

Here goes::

    gpu     - GPU vector engine.
    mcore   - Multicore vector engine.
    naive   - Single-core engine with a naive implementation of array traversal; compute_naive_* etc.
    simple  - Single-core vector engine with "fruit-loops" optimizations; compute_* etc.
    score   - Experiment with implementing cache-tiling based on compute_naive_* traversal.
    tile    - Single-core engine experimenting with implementing cache-tiling based on compute_* traversal.
    dynamite- JIT compiled engine for CPU, the first of its kind ;)

