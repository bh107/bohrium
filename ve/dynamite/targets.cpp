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
    virtual bool compile(const char* symbol, const char* sourcecode, size_t source_len) = 0;
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

    process(
        const char* process_str,
        const char* object_path,
        const char* kernel_path
    ) :
        handle(NULL), 
        process_str(process_str), 
        object_path(object_path),
        kernel_path(kernel_path)
    {}
    
    bool compile(const char* symbol, const char* sourcecode, size_t source_len)
    {
        if (handle) {           // Unload a previous compilation
            dlclose(handle);
            handle = NULL;
        }

        int lib_fd;             // Library file-descriptor
        int kernel_fd;          // Kernel file-descriptor
        FILE *lib_fp    = NULL; // Handle for library-file
        FILE *kernel_fp = NULL; // Handle for kernel-file
        char *error     = NULL; // Buffer for dlopen errors
        char cmd[200];

        char *kernel_fn = NULL; // Kernel filename (kernel/kernel_XXXXXX)
        assign_string(kernel_fn, kernel_path);
        kernel_fd = mkstemp(kernel_fn);                 // Write to file (sourcecode)
        if (!kernel_fd) {                               // For offline inspection
            std::cout << "Err: Failed creating temp kernel-file." << std::endl;
            free(kernel_fn);
            return false;
        }
        kernel_fp = fdopen(kernel_fd, "w+");
        if (!kernel_fp) {
            std::cout << "Err: Failed opening kernel-file for writing." << std::endl;
            return false;
        }
        fwrite(sourcecode, 1, source_len, kernel_fp);
        fflush(kernel_fp);
        fclose(kernel_fp);
        close(kernel_fd);

        char *lib_fn = NULL;                        // Library filename (objects/object_XXXXXX)
        assign_string(lib_fn, object_path);
        lib_fd = mkstemp(lib_fn);                   // Filename of .o (object-file)
        if (-1==lib_fd) {
            std::cout << "Err: Could not create lib-tmp-file!" << std::endl;
            free(lib_fn);
            return false;
        }
        close(lib_fd);                              // Close it immediatly.

        sprintf(cmd, "%s%s", process_str, lib_fn);  // Add .o file-name to command

        lib_fp = popen(cmd, "w");                       // Execute the command
        if (!lib_fp) {
            std::cout << "Err: Could not execute process!" << std::endl;
            return false;
        }
        fwrite(sourcecode, 1, source_len, lib_fp);      // Write to stdin (sourcecode)
        fflush(lib_fp);
        pclose(lib_fp);
                                                // Load the kernel
        handle = dlopen(lib_fn, RTLD_NOW);      // Load library
        if (!handle) {
            std::cout << "Err: dlopen() failed." << std::endl;
            return false;
        }

        dlerror();                              // Clear any existing error
                                                // Load function from lib
        f = (func)dlsym(handle, symbol);
        error = dlerror();
        if (error) {    // TODO: Fix this output... of error
            std::cout << "Failed loading function!" << error << std::endl;
            free(error);
            return false;
        }

        return true;
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

private:
    const char* object_path;
    const char* kernel_path;

};

#endif

