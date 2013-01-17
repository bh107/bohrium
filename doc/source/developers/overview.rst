.. _developer_overview:

Overview
========

...

Conventions
~~~~~~~~~~~

...

Core
~~~~

Instructions
~~~~~~~~~~~~

Think of instructions as good old MIPS-assembly::

  ADD $d, $s, $t

But instead of the operands $d, $s, and $t being registers they are either arrays, scalars or constants. Instructions are encapsulations of the most basic executable operation. For correct execution then the following set of statements must be true for any instance of a Bohrium instruction.

The instruction has:

  * At least two operands
  * At most three operands
  * At most one destination operand
  * The first operand is always the destination operand

Whether an operand is an array, a scalar or a constant is determined at runtime by inspecting an instance of the struct:

.. doxygenstruct:: bh_instruction
   :project: Bohrium
   :path: doxygen/xml

If operand[i] == NULL then the operand i is a constant and the value bh_instruction.constant is type cast to the appropriate type according to bh_instruction.constant_type and used as operand.

TODO: described instruction interpretation/handling, stuff like what is supposed to happen to inst.status.

Data Structures and Types
~~~~~~~~~~~~~~~~~~~~~~~~~

Scalars, constants and array elements all belong to one of the basic types defined bh_type.h.


Component Communication
~~~~~~~~~~~~~~~~~~~~~~~

...

Component Configuration
~~~~~~~~~~~~~~~~~~~~~~~

...

.. 


