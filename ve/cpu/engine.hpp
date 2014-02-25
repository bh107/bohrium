#ifndef __BH_VE_CPU_ENGINE
#define __BH_VE_CPU_ENGINE
#include <string>

#include "bh.h"
#include "bh_vcache.h"

#include "tac.h"
#include "block.hpp"
#include "store.hpp"
#include "compiler.hpp"
#include "specializer.hpp"

class Engine {

public:
    Engine(
        std::string compiler_cmd,
        std::string template_directory,
        std::string kernel_directory,
        std::string object_directory
    );

    Engine(
        std::string compiler_cmd,
        std::string template_directory,
        std::string kernel_directory,
        std::string object_directory,
        int vcache_size,
        bool preload,
        bool jit_enabled,
        bool jit_fusion,
        bool jit_optimize,
        bool jit_dumpsrc
    );

    ~Engine();

    std::string text();

    bh_error execute(bh_ir* ir);

private:
    std::string compiler_cmd,
                template_directory,
                kernel_directory,
                object_directory;

    size_t vcache_size;

    bool preload,
         jit_enabled,
         jit_fusion,
         jit_optimize,
         jit_dumpsrc;
    
    Compiler    compiler;
    Store       storage;
    Specializer specializer;

};

#endif