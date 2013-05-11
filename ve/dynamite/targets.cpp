#ifndef __BH_VE_DYNAMITE_BACKENDS
#define __BH_VE_DYNAMITE_BACKEND

#include <iostream>
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>

#include "utils.h"

typedef void (*func)(int tool, ...);

/**
 * The target interface.
 *
 * Becomes what it compiles.
 */
class target {
public:
    virtual int compile(const char* symbol, const char* sourcecode, size_t source_len) = 0;
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
class process: target {
public:
    func f; // This is what it is all about :)

    process(const char* process_str) : handle(NULL), process_str(process_str) {}
    
    int compile(const char* symbol, const char* sourcecode, size_t source_len)
    {
        if (handle) {
            dlclose(handle);
            handle = NULL;
        }

        int fd;                                 // Handle for tmp-object-file
        FILE *p;                                // Handle for process
        char lib_fn[]   = "objects/object_XXXXXX";
        char cmd[200];                          // Buffer for command-string
        char *error;

        fd = mkstemp(lib_fn);                   // Filename of object-file
        if (-1==fd) {
            std::cout << "Err: Could not create lib-tmp-file!" << std::endl;
            return 0;
        }
        close(fd);                              // Close it immediatly.

        sprintf(cmd, "%s%s", process_str, lib_fn);  // Merge command-string

        p = popen(cmd, "w");                    // Execute it
        if (!p) {
            std::cout << "Err: Could not execute process!" << std::endl;
            return 0;
        }
        fwrite(sourcecode, 1, source_len, p);
        fflush(p);
        pclose(p);
                                                // Load the kernel
        handle = dlopen(lib_fn, RTLD_NOW);      // Load library
        if (!handle) {
            std::cout << "Err: dlopen() failed." << std::endl;
            return 0;
        }

        dlerror();                              // Clear any existing error
                                                // Load function from lib
        f = (func)dlsym(handle, symbol);
        error = dlerror();
        if (error) {
            std::cout << "Failed loading function!" << error << std::endl;
            return 0;
        }

        return 1;
    }

    ~process()
    {
        if (handle) {
            dlclose(handle);
            handle = NULL;
        }
    }

protected:
    void* handle;
    const char* process_str;

};

#endif

