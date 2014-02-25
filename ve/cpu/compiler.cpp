#include "compiler.hpp"

using namespace std;

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
{   /*
    if (handle) {
        dlclose(handle);
        handle = NULL;
    }*/
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
 */
bool Compiler::compile(string symbol, string library, const char* sourcecode, size_t source_len)
{
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
    pclose(cmd_stdin);

    /*

    //
    // Update the library mapping such that a load for the symbol
    // can the resolve the library that it needs
    libraries.insert(
        pair<string, string>(symbol, library)
    );
    */
    return true;
}

