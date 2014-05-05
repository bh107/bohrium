#ifndef __BH_VE_CPU_COMPILER
#define __BH_VE_CPU_COMPILER

#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>

namespace bohrium{
namespace engine {
namespace cpu {

class Compiler {
public:
    Compiler(const std::string process_str);
    ~Compiler();

    std::string text();

    bool compile(
        std::string object_abspath,
        const char* sourcecode,
        size_t source_len
    );

private:
    std::string process_str;

    static const char TAG[];
};

}}}

#endif
