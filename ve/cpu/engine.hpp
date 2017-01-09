#ifndef __KP_ENGINE_ENGINE_HPP
#define __KP_ENGINE_ENGINE_HPP 1

#include <string>
#include <vector>
#include <map>
#include "kp.h"
#include "block.hpp"
#include "symbol_table.hpp"
#include "program.hpp"
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
        const size_t jit_offload,
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

    std::string text();

    /**
     *  Generate and compile source, construct Block(kp_block) for execution.
     *  Send the program on to the C Runtime for execution
     */
    bh_error process_block(core::Block &block);

private:
    kp_rt* rt_;

    bool preload_,
         jit_enabled_,
         jit_dumpsrc_,
         jit_fusion_,
         jit_contraction_;

    size_t jit_offload_;
    
    Store           storage_;
    codegen::Plaid  plaid_;
    Compiler        compiler_;

    static const char TAG[];
};

}}

#endif

