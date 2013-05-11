#ifndef __BH_VE_DYNAMITE_BACKENDS
#define __BH_VE_DYNAMITE_BACKEND

#include <iostream>
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <dlfcn.h>

#include "utils.h"

// How should these varying signatures be handled???
//typedef double (*func)(double x, double y);

/*
typedef void (*func)(int64_t a0_start, int64_t* a0_stride, float* a0_data,
              int64_t a1_start, int64_t* a1_stride, float* a1_data,
              int64_t a2_start, int64_t* a2_stride, float* a2_data,
              int64_t* shape,
              int64_t ndim,
              int64_t nelements);
              */

typedef void (*func)(int tool, ...);

/**
 * The backend interface.
 *
 * Becomes what it compiles.
 */
class backend {
public:
    virtual int compile(const char* sourcecode, size_t source_len) = 0;
    virtual double execute(double left, double right) = 0;
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
class process: backend {
public:
    process(const char* process_str) : handle(NULL), process_str(process_str) {}
    
    int compile(const char* sourcecode, size_t source_len)
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
            std::cout << "Failed creating lib-tmp-file!" << std::endl;
            return 0;
        }
        close(fd);                              // Close it immediatly.

        sprintf(cmd, "%s%s", process_str, lib_fn);  // Merge command-string

        p = popen(cmd, "w");                    // Execute it
        if (!p) {
            std::cout << "Failed executing process!" << std::endl;
            return 0;
        }
        fwrite(sourcecode, 1, source_len, p);
        fflush(p);
        pclose(p);
                                                // Load the kernel
        handle = dlopen(lib_fn, RTLD_NOW);      // Load library
        if (!handle) {
            std::cout << "Failed loading library!" << std::endl;
            return 0;
        }

        dlerror();                              // Clear any existing error
                                                // Load function from lib
        f = (func)dlsym(handle, "traverse_aaa");
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
    
    double execute(double left, double right)
    {
        return 0.0;
    };

    func f;

protected:
    void* handle;
    const char* process_str;

};

/*
int main()
{
    double res;

    //tccl bck = tccl();
    //process bck("tcc -O2 -march=core2 -fPIC -x c -shared - -o ");
    //process bck("gcc -O2 -march=core2 -fPIC -x c -shared - -o ");
    //opencl bck = opencl();
    process bck("clang -O2 -march=core2 -fPIC -x c -shared - -o ");
    
    char* sourcecode = NULL;
    size_t source_len = read_file("templates/kernel.c", &sourcecode);   // Read sourcecode
    if(!source_len) {
        cout << "Failed reading sourcecode!" << endl;
        return -1;
    }

    int k = 0;
    bck.compile(sourcecode, source_len);
    for(int i=0; i<100000; i++,k++) {
        res = bck.execute(2.0, 3.0);
    }

    free(sourcecode);
    std::cout << "RES=" << res << ". " << k <<std::endl;
    return 0;
}
*/

#endif

