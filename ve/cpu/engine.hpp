#ifndef __KP_ENGINE_ENGINE_HPP
#define __KP_ENGINE_ENGINE_HPP 1

#include <string>
#include <vector>
#include <map>
#include "bh.h"
#include "kp.h"
#include "block.hpp"
#include "symbol_table.hpp"
#include "program.hpp"
#include "accelerator.hpp"
#include "store.hpp"
#include "compiler.hpp"
#include "plaid.hpp"
#include "codegen.hpp"

namespace kp{
namespace engine{

class Engine {
public:
    Engine(
        const kp_thread_binding binding,
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

    size_t vcache_size(void);
    bool preload(void);
    bool jit_enabled(void);
    bool jit_dumpsrc(void);
    bool jit_fusion(void);
    bool jit_contraction(void);
    bool jit_offload(void);
    int jit_offload_devid(void);

    std::string text();

    /**
     *  Generate and compile source, construct Block(kp_block) for execution.
     */
    bh_error process_block(core::Program& tac_program,
                           core::SymbolTable &symbol_table,
                           core::Block &block);
    
    /**
     *  Execute the given Block(kp_block), that is, buffer management
     *  and possible execution of a kernel function.
     */
    bh_error execute_block(core::Program& tac_program,
                           core::SymbolTable &symbol_table,
                           core::Block &block,
                           kp_krnl_func func);
    
private:
    kp_rt* rt_;

    bool preload_,
         jit_enabled_,
         jit_dumpsrc_,
         jit_fusion_,
         jit_contraction_,
         jit_offload_;

    int jit_offload_devid_;
    
    Store           storage_;
    codegen::Plaid  plaid_;
    Compiler        compiler_;
    std::vector<Accelerator*>   accelerators_;

    static const char TAG[];
};

}}

#endif

