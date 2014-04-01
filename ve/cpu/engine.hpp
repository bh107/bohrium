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
    std::string symbol_table_text(std::string prefix);

    bh_error register_extension(bh_component& instance, const char* name, bh_opcode opcode);
    bh_error execute(bh_ir& ir);

private:

    size_t map_operand(bh_instruction& instr, size_t operand_idx);
    bh_error map_operands(bh_instruction* instr);

    bh_error sij_mode(Block& block);
    bh_error fuse_mode(Block& block);

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

    operand_t* symbol_table;
    size_t nsymbols;

    static const char TAG[];
};

}}}
#endif
