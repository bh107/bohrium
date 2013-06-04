#ifndef __BH_VE_DYNAMITE_BACKENDS
#define __BH_VE_DYNAMITE_BACKEND

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
#include "utils.cpp"
#include <fcntl.h>

// Create nice error-messages...
int error(int errnum, const char *fmt, ...) {
    va_list va;
    int ret;

    char err_msg[200];
    sprintf(err_msg, "Errno=[%d, %s]: %s", errnum, strerror(errnum), fmt);
    va_start(va, fmt);
    ret = vfprintf(stderr, err_msg, va);
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
        const char* process_str,
        const char* object_path,
        const char* kernel_path,
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
        if ((dir = opendir (object_path)) != NULL) {
            while ((ent = readdir (dir)) != NULL) {
                char object_name[200];
                size_t name_len = strlen(ent->d_name);
                if (10>name_len) {              // Not what we want
                    continue;
                }

                strncpy(object_name, ent->d_name, name_len);
                object_name[name_len]=0;
                if ((object_name[0] == 'B' && \
                     object_name[1] == 'H' && \
                     object_name[2] == '_')) {  // Load single symbol
                     
                    object_name[name_len-10] = 0;   // Strip off "_UID.so"
                    if (load(std::string(object_name))) {
                        ++nloaded;
                    };

                } else if ( (object_name[name_len-1] == 'd') && \
                            (object_name[name_len-2] == 'n') && \
                            (object_name[name_len-3] == 'i') && \
                            (object_name[name_len-4] == '.')) {

                    std::vector<std::string> symbols;
                    char basename[256],
                         library[256],
                         index[256];

                    strncpy(basename, object_name, name_len-4);
                    basename[name_len-4] = 0;
                    sprintf(library, "%s/%s.so", object_path, basename);
                    sprintf(index,   "%s/%s.ind", object_path, basename);

                    std::ifstream symbol_file(index);
                    for(std::string symbol; getline(symbol_file, symbol);) {
                        symbols.push_back(symbol);
                    }
                    symbol_file.close();

                    nloaded += load(symbols, library);

                } else {                        // Load multiple symbols
                    std::cout << "Ignorning non-loadable file: [" << object_name << "] found in object-path." << std::endl;
                }
            }
            closedir (dir);
            return nloaded;
        } else {
            throw std::runtime_error("Failed opening bla bla lba.");
        }
    }

    /**
     *  Load symbol into func-storage.
     */
    bool load(std::string symbol)
    {
        char *error     = NULL;     // Buffer for dlopen errors
        char lib_fn[250] = "";       // Library filename (objects/<symbol>_XXXXXX)
        sprintf(
            lib_fn, 
            "%s/%s_%s.so",
            object_path,
            symbol.c_str(),
            uid
        );     

        handles[symbol] = dlopen(lib_fn, RTLD_NOW); // Open library
        if (!handles[symbol]) {
            std::cout << "Err: dlopen() failed. Lib=["<< lib_fn <<"], Symbol=["<< symbol <<"]" << std::endl;
            return false;
        }

        dlerror();                                  // Clear any existing error
                                                    // Load function from library
        funcs[symbol] = (func)dlsym(handles[symbol], symbol.c_str());
        error = dlerror();
        if (error) {
            std::cout << "Err: Failed loading [" << symbol << "], error=[" << error << "]" << std::endl;
            free(error);
            return false;
        }
        return true;
    }

    /**
     *  Load symbol into func-storage.
     */
    bool load(std::vector<std::string> symbols, std::string library)
    {
        char *error     = NULL;     // Buffer for dlopen errors

        handles[library] = dlopen(library.c_str(), RTLD_NOW); // Open library
        if (!handles[library]) {
            std::cout << "Err: dlopen() failed. Lib=["<< library <<"], multiple symbols." << std::endl;
            return false;
        }

        for(std::vector<std::string>::iterator symbol=symbols.begin(); symbol != symbols.end(); ++symbol) {
            dlerror();                                  // Clear any existing error
                                                        // Load function from library
            funcs[*symbol] = (func)dlsym(handles[library], (*symbol).c_str());
            error = dlerror();
            if (error) {
                std::cout << "Err: Failed loading [" << *symbol << "], error=[" << error << "]" << std::endl;
                free(error);
                return false;
            }
        }
        return true;
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
        char kernel_fn[250] = "";   // TODO: Make sure this is not overflown
        const char *mode = "w";
        int err;

        sprintf(kernel_fn, "%s/%s_%s.c", kernel_path, symbol.c_str(), uid);
        kernel_fd = open(kernel_fn, O_WRONLY | O_CREAT | O_EXCL, 0644);
        if ((!kernel_fd) || (kernel_fd<1)) {
            err = errno;
            error(err, "Failed opening kernel-file [%s].\n", kernel_fn);
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
        char lib_fn[250] = "";          // Library filename (objects/<symbol>_XXXXXX)
        sprintf(
            lib_fn,
            "%s/%s_%s.so",
            object_path,
            library.c_str(),
            uid
        );

        /*
        if (access(lib_fn, F_OK) == 0) {    // Load object if it exists
            return load(library);
        }*/

        // WARN: These constants must be safeguarded... they will bite you at some point!
        FILE *cmd_stdin     = NULL;  // Handle for library-file
        char cmd[1000]      = "";    // Command-line for executing compiler
        sprintf(
            cmd, 
            "%s %s",
            process_str,
            lib_fn
        );      
        cmd_stdin = popen(cmd, "w");                    // Execute the command
        if (!cmd_stdin) {
            std::cout << "Err: Could not execute process! ["<< cmd <<"]" << std::endl;
            return false;
        }
        fwrite(sourcecode, 1, source_len, cmd_stdin);   // Write to stdin (sourcecode)
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

private:
    handle_storage handles;
    char uid[7];
    const char *process_str;
    const char* object_path;
    const char* kernel_path;

};

#endif

