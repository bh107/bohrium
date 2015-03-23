#ifndef __BH_VE_CPU_ENGINE
#define __BH_VE_CPU_ENGINE
#include "bh.h"
#include "bh_vcache.h"

#include "tac.h"
#include "block.hpp"
#include "symbol_table.hpp"
#include "thread_control.hpp"
#include "store.hpp"
#include "compiler.hpp"
#include "plaid.hpp"
#include "codegen.hpp"

#include <string>
#include <vector>
#include <map>

namespace bohrium{
namespace engine {
namespace cpu {

class Engine {
public:
    Engine(
        const std::string compiler_cmd,
        const std::string compiler_inc,
        const std::string compiler_lib,
        const std::string compiler_flg,
        const std::string compiler_ext,
        const std::string template_directory,
        const std::string kernel_directory,
        const std::string object_directory,
        const size_t vcache_size,
        const bool preload,
        const bool jit_enabled,
        const bool jit_fusion,
        const bool jit_dumpsrc,
        const bool dump_rep,
        const thread_binding binding,
        const size_t mthreads
        );

    ~Engine();

    std::string text();

    bh_error register_extension(bh_component& instance, const char* name, bh_opcode opcode);

    bh_error execute(bh_ir* bhir);

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
    bh_error sij_mode(core::SymbolTable& symbol_table, std::vector<tac_t>& program, core::Block& block);

    /**
     *  Compile and execute multiple tac/instructions at a time.
     *
     *  This execution mode is used when
     *
     *      - jit_fusion=true,
     *      - The block contains at least one array operation (should be increased to more than 1)
     *      - The block contains does not contain any extensions
     */
    bh_error fuse_mode(
        core::SymbolTable& symbol_table,
        std::vector<tac_t>& program,
        core::Block& block,
        bh_ir_kernel& krnl
    );

    std::string compiler_cmd,
                template_directory,
                kernel_directory,
                object_directory;

    size_t vcache_size;

    bool preload,
         jit_enabled,
         jit_fusion,
         jit_dumpsrc,
         dump_rep;
    
    Store           storage;
    codegen::Plaid  plaid_;
    Compiler        compiler;
    ThreadControl   thread_control;

    std::map<bh_opcode, bh_extmethod_impl> extensions;

    static const char TAG[];
    size_t exec_count;
};

}}}
#endif
