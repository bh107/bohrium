Frequently Asked Questions (FAQ)
================================

**Does it automatically support lazy evaluation (also called: late evaluation, expression templates)?**

Yes, Bohrium will lazy evaluate all Python/NumPy operations until it encounters a "Python Read", such a printing an array or having an if-statement testing the value of an array.

**Does it support "views" in the sense that a sub-slice is simply a view  on the same array?**

Yes, Bohrium supports NumPy views fully thus operating on array slices does not involve data copying.

**Does it support generator functions (which only start calculating once the evaluation is forced)? Which ones are supported?  Which conditions force evaluations? Presumably reduce operations?**

Yes, Bohrium uses a fusion algorithm that fuses (or merges) array operations into the same computation kernel that are then JIT-compiled and executed. However, Bohrium can only fuse operations that have some common sized dimension and no horizontal data conflicts. Typically, reducing a vector to a scalar will force evaluate (but reducing a matrix to a vector will not force an evaluate on it own).


**On GPUs, will Bohrium automatically keep all data (i.e. all Bohrium arrays) on the card?**

Yes, we only move data back to the host when the data is accessed directly by Python or a Python C-extension.

**Does it fully support operations on the complex datatype in Bohrium arrays?**

Yes.

**Will it lazily operate even over for-loops effectively unrolling them?**

Yes, a for-loop in Python does not force evaluation. However, loops in Python with many iterations will hurt performance, just like it does in regular NumPy or Matlab


**Is Bohrium using CUDA on Nvidia Cards or generic OpenCL for any GPU?**

At the moment, Bohrium uses OpenCL for both Nvidia, AMD, and Intel graphic cards.

**What is the disadvantage of Bohrium? I wonder why it exists as a separate project. From my point of view it looks like Bohrium is "just reimplementing" NumPy. That's probably extremely oversimplified, but is there a plan to feed the results of Bohrium into the NumPy project?**

The only disadvantage of Bohrium is the extra dependencies e.g. Bohrium need a C99 compiler for JIT-complication. Thus, the idea of incorporating Bohrium into NumPy as an alternative "backend" is very appealing and we hope it could be realized some day.





