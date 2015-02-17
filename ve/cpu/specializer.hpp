#ifndef __BH_VE_CPU_SPECIALIZER
#define __BH_VE_CPU_SPECIALIZER
#include "block.hpp"
#include "utils.hpp"
#include "plaid.hpp"

#include <ctemplate/template.h>
#include <string>
#include <vector>

using namespace bohrium::core;
namespace bohrium {
namespace engine {
namespace cpu {

class Specializer {
public:
   
    Specializer(const std::string template_directory);
    ~Specializer();

    std::string text();

    std::string cexpression(    SymbolTable& symbol_table,
                                const Block& block,
                                size_t tac_idx);

    std::string template_filename(  SymbolTable& symbol_table,
                                    const Block& block,
                                    size_t pc);

    std::string specialize( SymbolTable& symbol_table,
                            Block& block);

    std::string specialize( SymbolTable& symbol_table,
                            Block& block,
                            size_t tac_start, size_t tac_end);

    std::string specialize( SymbolTable& symbol_table,
                            Block& block,
                            LAYOUT fusion_layout);

    codegen::Plaid plaid_;

private:
    ctemplate::Strip strip_mode;
    std::string template_directory;

    static const char TAG[];
};

}}}
#endif
