#ifndef __BH_VE_CPU_ENGINE
#define __BH_VE_CPU_ENGINE
#include "bh.h"
#include "bh_vcache.h"

#include "tac.h"
#include "block.hpp"
#include "store.hpp"
#include "compiler.hpp"
#include "specializer.hpp"

#include <string>

namespace bohrium{
namespace engine {
namespace cpu {

class Engine {

public:

    Engine(
        const std::string compiler_cmd,
        const std::string template_directory,
        const std::string kernel_directory,
        const std::string object_directory,
        const size_t vcache_size,
        const bool preload,
        const bool jit_enabled,
        const bool jit_fusion,
        const bool jit_optimize,
        const bool jit_dumpsrc);

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
    
    Store          storage;
    Specializer    specializer;
    Compiler       compiler;
};

}}}
#endif