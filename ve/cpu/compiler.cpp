#include "utils.hpp"
#include "compiler.hpp"
#include <sstream>
using namespace std;

namespace bohrium{
namespace engine {
namespace cpu {

const char Compiler::TAG[] = "Compiler";

Compiler::Compiler(string cmd, string inc, string lib, string flg, string ext) : cmd_(cmd), inc_(inc), lib_(lib), flg_(flg), ext_(ext) {}

Compiler::~Compiler() {}

string Compiler::text()
{
    stringstream ss;
    ss << "Compiler(\"" << process_str("OBJ", "SRC") << "\")";
    ss << endl;

    return ss.str();
}

string Compiler::process_str(string object_abspath, string source_abspath)
{
    stringstream ss;

    ss           << cmd_;
    ss << " "    << inc_; 
    ss << " "    << flg_;
    ss << " "    << ext_;
    ss << " "    << source_abspath;
    ss << " "    << lib_;
    ss << " -o " << object_abspath;

    return ss.str();
}

/**
 *  Compile the given sourcecode into a shared object.
 *
 *  @returns True the system compiler returns exit-code 0 and False othervise.
 */
bool Compiler::compile(string object_abspath, const char* sourcecode, size_t source_len)
{
    string cmd = process_str(object_abspath, " - ");

    FILE* cmd_stdin = popen(cmd.c_str(), "w");  // Open process and get stdin stream
    if (!cmd_stdin) {
        perror("popen()");
        fprintf(stderr, "popen() failed for: [%s]", sourcecode);
        pclose(cmd_stdin);
        return false;
    }
                                                // Write / pipe to stdin
    int write_res = fwrite(sourcecode, sizeof(char), source_len, cmd_stdin);
    if (write_res < (int)source_len) {
        perror("fwrite()");
        fprintf(stderr, "fwrite() failed in file %s at line # %d\n", __FILE__, __LINE__-5);
        pclose(cmd_stdin);
        return false;
    }

    int flush_res = fflush(cmd_stdin);          // Flush stdin
    if (EOF == flush_res) {
        perror("fflush()");
        fprintf(stderr, "fflush() failed in file %s at line # %d\n", __FILE__, __LINE__-5);
        pclose(cmd_stdin);
        return false;
    }

    int exit_code = (pclose(cmd_stdin)/256);
    if (0!=exit_code) {
        perror("pclose()");
        fprintf(stderr, "pclose() failed.\n");
    }
    
    return (exit_code==0);
}

/**
 *  Compile the given sourcecode into a shared object.
 *
 *  @returns True the system compiler returns exit-code 0 and False othervise.
 */
bool Compiler::compile(string object_abspath, string src_abspath)
{
    string cmd = process_str(object_abspath, src_abspath);

    // Execute the process
    FILE *cmd_stdin = NULL;                     // Handle for library-file
    cmd_stdin = popen(cmd.c_str(), "w");        // Execute the command
    if (!cmd_stdin) {
        std::cout << "Err: Could not execute process! ["<< cmd <<"]" << std::endl;
        return false;
    }

    fflush(cmd_stdin);
    int exit_code = (pclose(cmd_stdin)/256);
    return (exit_code==0);
}

}}}
