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
	/**
	 * compile() forks and executes a system process, the process along with
	 * arguments must be provided as argument at time of construction.
	 * The process must be able to consume sourcecode via stdin and produce
	 * a shared object file.
	 * The compiled shared-object is then loaded and made available for execute().
	 *
	 * Examples:
	 *
	 *  Compiler tcc("tcc", "", "-lm", "-O2 -march=core2", "-fPIC -x c -shared");
	 *  Compiler icc("ic",  "", "-lm", "-O2 -march=core2", "-fPIC -x c -shared");
	 *  Compiler gcc("gcc", "", "-lm", "-O2 -march=core2", "-fPIC -x c -shared");
	 *
	 */
    Compiler(std::string cmd, std::string inc, std::string lib, std::string flg, std::string ext);
    ~Compiler();

    std::string text();
    std::string process_str(std::string object_abspath, std::string source_abspath);

    /**
     *  Compile by piping sourcecode to stdin.
     */
    bool compile(
        std::string object_abspath,
        const char* sourcecode,
        size_t source_len
    );

    /**
     *  Compile source on disk.
     */
    bool compile(
        std::string object_abspath,
        std::string src_abspath
    );


private:
    std::string cmd_, inc_, lib_, flg_, ext_;

    static const char TAG[];
};

}}}

#endif
