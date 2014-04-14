#ifndef __BH_VE_CPU_SPECIALIZER
#define __BH_VE_CPU_SPECIALIZER
#include "block.hpp"
#include "utils.hpp"

#include <ctemplate/template.h>
#include <string>
#include <vector>

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
                            const Block& block,
                            std::vector<triplet_t>& ranges);

    std::string specialize( SymbolTable& symbol_table,
                            const Block& block);

    std::string specialize( SymbolTable& symbol_table,
                            const Block& block,
                            size_t tac_start, size_t tac_end);

private:
    ctemplate::Strip strip_mode;
    std::string template_directory;

    static const char TAG[];
};

}}}
#endif
