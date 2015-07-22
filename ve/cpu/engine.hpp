#ifndef __BH_VE_CPU_ENGINE
#define __BH_VE_CPU_ENGINE
#include "bh.h"
#include "bh_vcache.h"

#include "tac.h"
#include "block.hpp"
#include "symbol_table.hpp"
#include "thread_control.hpp"
#include "accelerator.hpp"
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
        const thread_binding binding,
        const size_t thread_limit,
        const size_t vcache_size,
        const bool preload,
        const bool jit_enabled,
        const bool jit_dumpsrc,
        const bool jit_fusion,
        const bool jit_contraction,
        const bool jit_offload,
        const std::string compiler_cmd,
        const std::string compiler_inc,
        const std::string compiler_lib,
        const std::string compiler_flg,
        const std::string compiler_ext,
        const std::string object_directory,
        const std::string template_directory,
        const std::string kernel_directory
    );

    ~Engine();

    std::string text();

    bh_error register_extension(bh_component& instance, const char* name, bh_opcode opcode);

    bh_error execute(bh_ir* bhir);

private:

    /**
     *  Compile and execute the given program.
     *
     */
    bh_error execute_block(
        core::SymbolTable& symbol_table,
        std::vector<tac_t>& program,
        core::Block& block,
        bh_ir_kernel& krnl
    );

    size_t vcache_size_;

    bool preload_,
         jit_enabled_,
         jit_dumpsrc_,
         jit_fusion_,
         jit_contraction_,
         jit_offload_;
    
    Store           storage_;
    codegen::Plaid  plaid_;
    Compiler        compiler_;
    ThreadControl   thread_control_;
    Accelerator     accelerator_;

    std::map<bh_opcode, bh_extmethod_impl> extensions_;

    static const char TAG[];
    size_t exec_count;
};

}}}
#endif
