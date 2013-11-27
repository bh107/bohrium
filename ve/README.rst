===================================
Brief Description of Vector Engines
===================================

Here goes::

    gpu - GPU vector engine.
    cpu - JIT compiled engine for CPU, the first of its kind ;)

    static/score    - CPU vector engine.
    static/mcore    - CPU vector engine with TLP.
    static/tiling   - CPU vector engine with tiling.
    shared/         - Objects / tools shared between engines
    shared/bundler  - Instruction bundler mainly for tiling engine.
    shared/compute  - Compute/dispatch functions used by gpu and static engines.

