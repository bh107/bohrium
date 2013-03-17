Bohrium C++ Vector Library / Language Bridge
============================================

Welcome!

You have found the exclusive home of the c++ language bridge.
A list of inhabitants (along with brief descriptions) of this part of the repository is provided below::

    bh/                 - The Bohrium c++ language-bridge / vector library.
    bh/cppb.hpp         - Declarations.
    bh/multi_array.hpp  - Operand definition.
    bh/functions.hpp    - Operations via functions.
                          Note: Generated code -> see codegen.
    bh/operators.hpp    - Operations via operator-overloads.
                          Note: Generated code -> see codegen.
    bh/runtime.hpp      - Communication with Bohrium runtime
    bh/traits.hpp       - Template-traits for assigning type to constants and arrays.
                          Note: Generated code -> see codegen.

    examples/               - Examples of using the Bohrium c++ library.
    examples/operators.cpp  - Shows how the operator overloads operates.
    examples/hw.cpp         - Simples example of using the bridge.

    bin/    - Compiled examples and tests goes here

    codegen/                    - Home of the code-generator.
    codegen/operators.json      - A mapping between C++ operators and Bohrium operators.
    codegen/element_types.json  - A mapping between bohrium types and C++ primitive types.
    codegen/
    codegen/gen.py              - Code generator script; does the actual code-generation.
    codegen/output/             - Temporary storage for generated code.
    codegen/templates/          - Cheetah string templates for code generator.

    Makefile    - Builds examples and runs code-generator.

    README.rst  - This file.

