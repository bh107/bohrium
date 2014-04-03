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

    std::string cexpression(const Block& block, size_t tac_idx);

    std::string template_filename(const Block& block, size_t pc);

    std::string specialize(const Block& block, bool apply_fusion);
    std::string specialize(const Block& block, size_t tac_start, size_t tac_end, bool apply_fusion);

private:
    ctemplate::Strip strip_mode;
    std::string template_directory;

    static const char TAG[];
};

}}}
#endif
