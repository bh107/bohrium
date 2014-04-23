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
#include <map>

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
        const bool jit_dumpsrc);

    ~Engine();

    std::string text();

    bh_error register_extension(bh_component& instance, const char* name, bh_opcode opcode);
    bh_error execute(bh_ir& ir);

private:
    /**
     *  Compile and execute the given block one tac/instruction at a time.
     *
     *  This execution mode is used when for one reason or another want to
     *  do interpret the execution instruction-by-instruction.
     *
     *  This will happen when
     *  
     *  The block does not contain array operations
     *  The block does contain array operations but also an extension
     *
     */
    bh_error sij_mode(SymbolTable& symbol_table, Block& block);

    /**
     *  Compile and execute multiple tac/instructions at a time.
     *
     *  This execution mode is used when
     *
     *      - jit_fusion=true,
     *      - The block contains at least one array operation (should be increased to more than 1)
     *      - The block contains does not contain any extensions
     */
    bh_error fuse_mode(SymbolTable& symbol_table, Block& block);

    std::string compiler_cmd,
                template_directory,
                kernel_directory,
                object_directory;

    size_t vcache_size;

    bool preload,
         jit_enabled,
         jit_fusion,
         jit_dumpsrc;
    
    Store          storage;
    Specializer    specializer;
    Compiler       compiler;

    std::map<bh_opcode, bh_extmethod_impl> extensions;

    static const char TAG[];
    size_t exec_count;
};

}}}
#endif
