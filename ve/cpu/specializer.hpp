#ifndef __BH_VE_CPU_SPECIALIZER
#define __BH_VE_CPU_SPECIALIZER
#include <string>

#include <ctemplate/template.h>
#include "block.hpp"

class Specializer {
public:

    Specializer(std::string kernel_directory, ctemplate::Strip strip_mode);
    Specializer(std::string kernel_directory);
    std::string template_filename(Block& block, int pc, int64_t optimized);
    std::string specialize(Block& block, int64_t optimized);

private:
    ctemplate::Strip strip_mode;

};

#endif