===================================
Brief Description of Vector Engines
===================================

Here goes::

    gpu      - GPU vector engine.
    naive    - Single-core engine with a naive implementation of array traversal; compute_naive_* etc.
    simple   - Single-core vector engine with "fruit-loops" optimizations; compute_* etc.
    dynamite - JIT compiled engine for CPU, the first of its kind ;)
    print    - prints the bytecode stream

