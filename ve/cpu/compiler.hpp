#ifndef __BH_VE_CPU_COMPILER
#define __BH_VE_CPU_COMPILER

#include <string>
#include <cstdio>
#include <iostream>

class Compiler {
public:
    Compiler(std::string process_str, std::string object_directory);
    ~Compiler();

    bool compile(
        std::string symbol,
        std::string library,
        const char* sourcecode,
        size_t source_len
    );

private:
    std::string process_str;
    std::string object_directory;

};

#endif