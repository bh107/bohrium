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

    std::string tac_cexpr(tac& tac, Block& block);

    std::string template_filename(Block& block, int pc, int64_t optimized);

    std::string specialize(Block& block, int64_t optimized);

private:
    ctemplate::Strip strip_mode;
    std::string template_directory;

};

}}}
#endif