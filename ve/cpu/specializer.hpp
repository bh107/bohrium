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
    ~Specializer();

    std::string text();

    std::string cexpression(Block& block, size_t tac_idx);

    std::string template_filename(Block& block, size_t pc, bool optimized);

    std::string specialize(Block& block, bool optimized);
    std::string specialize(Block& block, bool optimized, size_t tac_start, size_t tac_end);

    std::string fuse(Block& block, bool optimized, size_t tac_start, size_t tac_end);

private:
    ctemplate::Strip strip_mode;
    std::string template_directory;

};

}}}
#endif
