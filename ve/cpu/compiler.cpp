#include "utils.hpp"
#include "compiler.hpp"

using namespace std;

namespace bohrium{
namespace engine {
namespace cpu {

const char Compiler::TAG[] = "Compiler";

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
Compiler::Compiler(string process_str) : process_str(process_str) {}

Compiler::~Compiler()
{
    DEBUG(TAG, "~Compiler()");
    DEBUG(TAG, "~Compiler()");
}

string Compiler::text()
{
    stringstream ss;
    ss << "Compiler(\"" << process_str << "\")";
    ss << endl;

    return ss.str();
}

/**
 *  Compile the given sourcecode into a shared object.
 *
 *  @returns True the system compiler returns exit-code 0 and False othervise.
 */
bool Compiler::compile(string object_abspath, const char* sourcecode, size_t source_len)
{
    DEBUG(TAG, "compile(" << object_abspath<< ", ..., ..." << ");");
    //
    // Constuct the compiler command
    string cmd = process_str +" "+ object_abspath;

    // Execute the process
    FILE *cmd_stdin = NULL;                     // Handle for library-file
    cmd_stdin = popen(cmd.c_str(), "w");        // Execute the command
    if (!cmd_stdin) {
        std::cout << "Err: Could not execute process! ["<< cmd <<"]" << std::endl;
        return false;
    }
    fwrite(sourcecode, 1, source_len, cmd_stdin);   // Write sourcecode to stdin
    fflush(cmd_stdin);
    int exit_code = (pclose(cmd_stdin)/256);
    DEBUG(TAG, "compile(...) : exit_code("<< exit_code << ");");
    return (exit_code==0);
}

}}}
