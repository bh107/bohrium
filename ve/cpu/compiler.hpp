#ifndef __BH_VE_CPU_COMPILER
#define __BH_VE_CPU_COMPILER

#include <string>
#include <sstream>
#include <cstdio>
#include <iostream>

class Compiler {
public:
    Compiler(const std::string process_str,
             const std::string object_directory);
    
    ~Compiler();

    std::string text();

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