#ifndef __BH_VE_CPU_SPECIALIZER
#define __BH_VE_CPU_SPECIALIZER
#include "block.hpp"
#include "utils.hpp"

#include <ctemplate/template.h>
#include <string>

namespace bohrium {
namespace engine {
namespace cpu {

class Specializer {
public:
   
    Specializer(const std::string template_directory);

    std::string text();

    std::string tac_operator_cexpr(Block& block, size_t tac_idx);

    std::string template_filename(Block& block, size_t pc, bool optimized);

    std::string specialize(Block& block, bool optimized);

private:
    ctemplate::Strip strip_mode;
    std::string template_directory;

};

}}}
#endif