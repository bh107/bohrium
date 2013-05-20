#ifndef __BH_VE_DYNAMITE_BACKENDS
#define __BH_VE_DYNAMITE_BACKEND

#include <iostream>
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>
#include <map>
#include "dirent.h"
#include <dlfcn.h>
#include "utils.cpp"
#include <fcntl.h>

typedef void (*func)(int tool, ...);
typedef std::map<std::string, func> func_storage;
typedef std::map<std::string, void*> handle_storage;

/**
 *  TODO: Load existing objects at startup.
 *          Then pre-compilation and warmup rounds will be possible.
 */

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
    {
        // Create an identifier with low collision...
        static const char alphanum[] = 
            "0123456789"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz";

        srand(getpid());
        for (int i = 0; i < 7; ++i) {
            uid[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
        }
        uid[6] = 0;
    }

    bool symbol_ready(std::string symbol) {
        return funcs.count(symbol) > 0;
    }

    bool load(std::string symbol)
    {
        if (funcs.count(symbol)>0) {
            return true;
        }

        char *error     = NULL;     // Buffer for dlopen errors
        char lib_fn[50] = "";       // Library filename (objects/<symbol>_XXXXXX)
        sprintf(
            lib_fn, 
            "%s%s_%s.so",
            object_path,
            symbol.c_str(),
            uid
        );     

        handles[symbol] = dlopen(lib_fn, RTLD_NOW); // Open library
        if (!handles[symbol]) {
            std::cout << "Err: dlopen() failed." << std::endl;
            return false;
        }

        dlerror();                              // Clear any existing error
                                                // Load function from library
        funcs[symbol] = (func)dlsym(handles[symbol], symbol.c_str());
        error = dlerror();
        if (error) {
            std::cout << "Err: Failed loading '" << symbol << "', error=['" << error << "']" << std::endl;
            free(error);
            return false;
        }

        return true;
    }

    /**
     *  Write source-code to file.
     *  Filename will be along the lines of: kernel/<symbol>_<UID>.c
     */
    bool src_to_file(std::string symbol, const char* sourcecode, size_t source_len)
    {
        int kernel_fd;              // Kernel file-descriptor
        FILE *kernel_fp = NULL;     // Handle for kernel-file
        char kernel_fn[50] = "";    // TODO: Make sure this is not overflown

        sprintf(kernel_fn, "%s%s_%s.c", kernel_path, symbol.c_str(), uid);
        kernel_fd = open(kernel_fn, O_WRONLY | O_CREAT | O_EXCL, 0644);
        if (!kernel_fd) {                               
            std::cout << "Err: Failed opening kernel-file << " << kernel_fn << "." << std::endl;
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

        return true;
    }

    bool compile(std::string symbol, const char* sourcecode, size_t source_len)
    {
        if (funcs.count(symbol)>0) {
            return true;
        }

        // WARN: These constants must be safeguarded... they will bite you at some point!
        FILE *cmd_stdin    = NULL;  // Handle for library-file
        char cmd[200]      = "";    // Command-line for executing compiler
        sprintf(
            cmd, 
            "%s %s%s_%s.so",
            process_str, object_path,
            symbol.c_str(),
            uid
        );      
        cmd_stdin = popen(cmd, "w");                    // Execute the command
        if (!cmd_stdin) {
            std::cout << "Err: Could not execute process!" << std::endl;
            return false;
        }
        fwrite(sourcecode, 1, source_len, cmd_stdin);   // Write to stdin (sourcecode)
        fflush(cmd_stdin);
        pclose(cmd_stdin);

        if (!src_to_file(symbol, sourcecode, source_len)) {
            return false;
        }

        if (!load(symbol)) {
            return false;
        }

        return true;
    }

    void load_symbols()
    {

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
    char uid[7];
    const char *process_str;
    const char* object_path;
    const char* kernel_path;

};

#endif

