#ifndef __BH_VE_DYNAMITE_BACKENDS
#define __BH_VE_DYNAMITE_BACKEND

#include <iostream>
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <map>
#include <dlfcn.h>
#include "utils.cpp"

typedef void (*func)(int tool, ...);
typedef std::map<std::string, func> func_storage;
typedef std::map<std::string, void*> handle_storage;

/**
 * The compiler interface.
 *
 * Becomes what it compiles.
 */
class compiler {
public:
    virtual bool compile(std::string symbol, const char* sourcecode, size_t source_len) = 0;
};

/**
 * compile() forks and executes a system process, the process along with
 * arguments must be provided as argument at time of construction.
 * The process must be able to consume sourcecode via stdin and produce
 * a shared object file.
 * The compiled shared-object is then loaded and made available for execute().
 *
 * Examples:
 *
 *  process tcc("tcc -O2 -march=core2 -fPIC -x c -shared - -o ");
 *  process gcc("gcc -O2 -march=core2 -fPIC -x c -shared - -o ");
 *  process clang("clang -O2 -march=core2 -fPIC -x c -shared - -o ");
 *
 */
class process: compiler {
public:
    func_storage funcs;

    process(
        const char* process_str,
        const char* object_path,
        const char* kernel_path
    ) :
        process_str(process_str), 
        object_path(object_path),
        kernel_path(kernel_path)
    {}

    bool symbol_ready(std::string symbol) {
        return funcs.count(symbol) > 0;
    }

    bool compile(std::string symbol, const char* sourcecode, size_t source_len)
    {
        if (funcs.count(symbol)>0) {
            return true;
        }

        int lib_fd;                 // Library file-descriptor
        FILE *lib_fp    = NULL;     // Handle for library-file

        int kernel_fd;              // Kernel file-descriptor
        FILE *kernel_fp = NULL;     // Handle for kernel-file

        char *error     = NULL;     // Buffer for dlopen errors

        // WARN: These constants must be safeguarded... they will bite you at some point!
        char cmd[200]      = "";    // Command-line for executing compiler
        char lib_fn[50]    = "";    // Library filename (objects/<symbol>_XXXXXX)
        char kernel_fn[50] = "";    // Kernel filename (kernel/<symbol>_XXXXXX)

        // Kernel file-name along the lines of: kernel/BH_ADD_DDD_FFF_nmL78p
        sprintf(kernel_fn, "%s%s_XXXXXX", kernel_path, symbol.c_str());
        kernel_fd = mkstemp(kernel_fn);                 // Write to file (sourcecode)
        if (!kernel_fd) {                               // For offline inspection
            std::cout << "Err: Failed creating temp kernel-file." << std::endl;
            return false;
        }
        kernel_fp = fdopen(kernel_fd, "w");
        if (!kernel_fp) {
            std::cout << "Err: Failed opening kernel-file for writing." << std::endl;
            return false;
        }
        fwrite(sourcecode, 1, source_len, kernel_fp);
        fflush(kernel_fp);
        fclose(kernel_fp);
        close(kernel_fd);

        // Library file-name along the lines of: object/BH_ADD_DDD_FFF_9Z61yH
        sprintf(lib_fn, "%s%s_XXXXXX", object_path, symbol.c_str());
        lib_fd = mkstemp(lib_fn);                       // Expand XXXXXX
        if (-1==lib_fd) {
            std::cout << "Err: Could not create lib-tmp-file!" << std::endl;
            return false;
        }
        close(lib_fd);                                  // Close it immediatly.

        // Command-line
        sprintf(cmd, "%s%s", process_str, lib_fn);      // Add .o file-name to command
        lib_fp = popen(cmd, "w");                       // Execute the command
        if (!lib_fp) {
            std::cout << "Err: Could not execute process!" << std::endl;
            return false;
        }
        fwrite(sourcecode, 1, source_len, lib_fp);      // Write to stdin (sourcecode)
        fflush(lib_fp);
        pclose(lib_fp);
        
        // Load the compiled kernel from the compiled library
        handles[symbol] = dlopen(lib_fn, RTLD_NOW);      // Open library
        if (!handles[symbol]) {
            std::cout << "Err: dlopen() failed." << std::endl;
            return false;
        }

        dlerror();                              // Clear any existing error
        funcs[symbol] = (func)dlsym(handles[symbol], symbol.c_str());        // Load function from lib
        error = dlerror();
        if (error) {
            std::cout << "Err: Failed loading'" << symbol << "', error=]" << std::endl;
            free(error);
            return false;
        }

        return true;
    }

    ~process()
    {   /*
        if (handle) {
            dlclose(handle);
            handle = NULL;
        }*/
    }

private:
    handle_storage handles;

    const char *process_str;

    const char* object_path;
    const char* kernel_path;

};

#endif

