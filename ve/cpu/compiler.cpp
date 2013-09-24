#ifndef __BH_VE_CPU_BACKENDS
#define __BH_VE_CPU_BACKENDS

#include <iostream>
#include <cstring>
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <unordered_map>
#include "dirent.h"
#include <dlfcn.h>
#include <unistd.h>
#include "utils.cpp"
#include <fcntl.h>

// Create nice error-messages...
int error(int errnum, const char *fmt, ...) {
    va_list va;
    int ret;

    char err_msg[500];
    sprintf(err_msg, "Error[%d, %s] from: %s", errnum, strerror(errnum), fmt);
    va_start(va, fmt);
    ret = vfprintf(stderr, err_msg, va);
    va_end(va);
    return ret;
}

int error(const char *err_msg, const char *fmt, ...) {
    va_list va;
    int ret;

    char err_txt[500];
    sprintf(err_txt, "Error[%s] from: %s", err_msg, fmt);
    va_start(va, fmt);
    ret = vfprintf(stderr, err_txt, va);
    va_end(va);
    return ret;
}

typedef void (*func)(int tool, ...);
//typedef std::map<std::string, func> func_storage;
//typedef std::map<std::string, void*> handle_storage;

typedef std::unordered_map<std::string, func> func_storage;
typedef std::unordered_map<std::string, void*> handle_storage;

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
        std::string process_str,
        std::string object_path,
        std::string kernel_path,
        bool do_preload
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
            uid[i] = 'a';
        }
        uid[6] = 0;

        if (do_preload) {     // Now load all objects...
            preload();      
        }
    }

    bool symbol_ready(std::string symbol) {
        return funcs.count(symbol) > 0;
    }

    size_t preload()
    {
        DIR *dir;
        struct dirent *ent;
        size_t nloaded = 0;
        if ((dir = opendir (object_path.c_str())) != NULL) {
            while ((ent = readdir (dir)) != NULL) {
                size_t fn_len = strlen(ent->d_name);

                if (14>fn_len) {              // Not what we want
                    continue;
                }

                std::string fn(ent->d_name),
                            lib_fn;

                if (0==fn.compare(0,3, "BH_")) {        // Single
                    lib_fn.assign(fn, 0, fn_len-10);    // Remove "_xxxxxx.so"
                    if (load(lib_fn, lib_fn)) {
                        ++nloaded;
                    };                                  // Multiple
                } else if (0==fn.compare(fn_len-4, 4, ".ind")) {
                    lib_fn.assign(fn, 0, fn_len-11);    // Remove "_xxxxxx.ind"
                    std::string index_fn = lib_path(lib_fn.c_str(), "ind");

                    std::vector<std::string> symbols;
                    std::ifstream symbol_file(index_fn);
                    for(std::string symbol; getline(symbol_file, symbol);) {
                        symbols.push_back(symbol);
                    }
                    symbol_file.close();

                    nloaded += load(symbols, lib_fn);

                } else {                                        // Ignore
                    std::cout << "Ignorning non-loadable file: ";
                    std::cout << "[" << fn << "] ";
                    std::cout << "found in object-path." << std::endl;
                }
            }
            closedir (dir);
            return nloaded;
        } else {
            throw std::runtime_error("Failed opening bla bla lba.");
        }
    }

    /**
     *  Load a single symbol from library symbol into func-storage.
     */
    bool load(std::string symbol, std::string library)
    {
        char *error_msg = NULL;             // Buffer for dlopen errors
        int errnum = 0;

        std::string library_fn = lib_path(  // "./objects/<symbol>_XXXXXX"
                library.c_str(),
                "so"
        );

        if (0==handles.count(library)) {    // Open library
            handles[library] = dlopen(          
                library_fn.c_str(),
                RTLD_NOW
            );
            errnum = errno;
        }
        if (!handles[library]) {            // Check that it opened
            error(
                errnum,
                "Failed openening library; dlopen(filename='%s', RTLF_NOW) failed.",
                library_fn.c_str()
            );
            return false;
        }

        dlerror();                          // Clear any existing error then,
        funcs[symbol] = (func)dlsym(        // Load symbol/function
            handles[library],
            symbol.c_str()
        );
        error_msg = dlerror();
        if (error_msg) {
            error(
                error_msg,
                "dlsym( handle='%s', symbol='%s' )\n",
                library_fn.c_str(),
                symbol.c_str()
            );
            free(error_msg);
            return false;
        }
        return true;
    }

    /**
     *  Load multiple symbols from library into func-storage.
     */
    bool load(std::vector<std::string> symbols, std::string library)
    {
        bool res = true;
        for(std::vector<std::string>::iterator symbol=symbols.begin();
            (symbol != symbols.end()) && res;
            ++symbol
        ) {
            res *= load(*symbol, library);
        }
        return res;
    }

    /**
     *  Write source-code to file.
     *  Filename will be along the lines of: kernel/<symbol>_<UID>.c
     *  NOTE: Does not overwrite existing files.
     */
    bool src_to_file(std::string symbol, const char* sourcecode, size_t source_len)
    {
        int kernel_fd;              // Kernel file-descriptor
        FILE *kernel_fp = NULL;     // Handle for kernel-file
        const char *mode = "w";
        int err;
        std::string kernel_fn = krn_path(symbol.c_str(), ".c");
        kernel_fd = open(kernel_fn.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        if ((!kernel_fd) || (kernel_fd<1)) {
            err = errno;
            error(err, "Failed opening kernel-file [%s] in src_to_file(...).\n", kernel_fn.c_str());
            return false;
        }
        kernel_fp = fdopen(kernel_fd, mode);
        if (!kernel_fp) {
            err = errno;
            error(err, "fdopen(fildes= %d, flags= %s).", kernel_fd, mode);
            return false;
        }
        fwrite(sourcecode, 1, source_len, kernel_fp);
        fflush(kernel_fp);
        fclose(kernel_fp);
        close(kernel_fd);

        return true;
    }

    bool compile(std::string library, const char* sourcecode, size_t source_len)
    {
        std::string cmd = command(library.c_str(), "so");
        FILE *cmd_stdin     = NULL;                     // Handle for library-file
        cmd_stdin = popen(cmd.c_str(), "w");            // Execute the command
        if (!cmd_stdin) {
            std::cout << "Err: Could not execute process! ["<< cmd <<"]" << std::endl;
            return false;
        }
        fwrite(sourcecode, 1, source_len, cmd_stdin);   // Write sourcecode to stdin
        fflush(cmd_stdin);
        pclose(cmd_stdin);

        return true;
    }

    ~process()
    {   /*
        if (handle) {
            dlclose(handle);
            handle = NULL;
        }*/
    }

    const char* get_uid(void)
    {
        return uid;
    }

    std::string lib_path(const char *lib_name, const char *ext)
    {
        return  object_path + "/" +\
                std::string(lib_name)    + "_" +\
                std::string(get_uid())   + "." +\
                std::string(ext);
    }

    std::string krn_path(const char *krn_name, const char *ext)
    {
        return  kernel_path + "/" +\
                std::string(krn_name)    + "_" +\
                std::string(get_uid())   + "." +\
                std::string(ext);
    }

    std::string command(const char *lib_name, const char *ext)
    {
        return  process_str + " "+\
                object_path + "/" +\
                std::string(lib_name)    + "_" +\
                std::string(get_uid())   + "." +\
                std::string(ext);
    }

private:
    handle_storage handles;
    char uid[7];
    std::string process_str;
    std::string object_path;
    std::string kernel_path;

};

#endif

