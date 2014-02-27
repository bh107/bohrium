#include "utils.hpp"
#include "compiler.hpp"

using namespace std;

namespace bohrium{
namespace engine {
namespace cpu {

/**
 * compile() forks and executes a system process, the process along with
 * arguments must be provided as argument at time of construction.
 * The process must be able to consume sourcecode via stdin and produce
 * a shared object file.
 * The compiled shared-object is then loaded and made available for execute().
 *
 * Examples:
 *
 *  Compiler tcc("tcc -O2 -march=core2 -fPIC -x c -shared - -o ");
 *  Compiler gcc("gcc -O2 -march=core2 -fPIC -x c -shared - -o ");
 *  Compiler clang("clang -O2 -march=core2 -fPIC -x c -shared - -o ");
 *
 */
Compiler::Compiler(string process_str, string object_directory ) : 
    process_str(process_str), object_directory(object_directory)
{}

Compiler::~Compiler()
{
    DEBUG(">>Compiler::~Compiler()");
    DEBUG("<<Compiler::~Compiler()");
}

string Compiler::text()
{
    
    stringstream ss;
    ss << "Compiler(\"" << process_str << "\", ";
    ss << "\"" << object_directory << "\");";
    ss << endl;

    return ss.str();
}

/**
 *  Compile a shared library for the given symbol.
 *  The library name is constructed using the uid of the process.
 *
bool Compiler::compile(string symbol, const char* sourcecode, size_t source_len)
{
    string library = symbol + "_" + string(get_uid());
    return compile(symbol, library, sourcecode, source_len);
}*/

/**
 *  Compile a shared library for the given symbol.
 *
 *  @returns True the system compiler returns exit-code 0 and False othervise.
 */
bool Compiler::compile(string symbol, string library, const char* sourcecode, size_t source_len)
{
    DEBUG(">> Compiler::compile(" << symbol << ", " << library << ", ..., ..." << ");");
    //
    // Constuct the compiler command
    string cmd = process_str +" "+ object_directory +"/"+ library +".so";

    // Execute it
    FILE *cmd_stdin = NULL;                     // Handle for library-file
    cmd_stdin = popen(cmd.c_str(), "w");        // Execute the command
    if (!cmd_stdin) {
        std::cout << "Err: Could not execute process! ["<< cmd <<"]" << std::endl;
        return false;
    }
    fwrite(sourcecode, 1, source_len, cmd_stdin);   // Write sourcecode to stdin
    fflush(cmd_stdin);
    int exit_code = (pclose(cmd_stdin)/256);
    DEBUG("<< Compiler::compile(...) : exit_code("<< exit_code << ");");
    return (exit_code==0);
}

}}}
