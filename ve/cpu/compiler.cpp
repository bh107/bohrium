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
#include <map>
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

//typedef void (*func)(int tool, ...);
typedef void (*func)(bh_kernel_arg_t args[]);

//
//  Retrieve a function pointer for the symbol (SYMBOL -> func)
typedef std::map<std::string, func> func_storage;

//
//  Retrieve the library handle for a library (LIBRARY -> handle)
typedef std::map<std::string, void*> handle_storage;

//
//  In which library is the symbol stored (SYMBOL -> LIBRARY)
typedef std::map<std::string, std::string> symbol_library_map;

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
class Compiler {
public:
    func_storage funcs;

    Compiler(
        std::string process_str,
        std::string object_dir,
        std::string kernel_dir,
        bool do_preload
    ) :
        process_str(process_str),
        object_dir(object_dir),
        kernel_dir(kernel_dir)
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

        if (do_preload) {   // Load whatever may be in the object-folder.
            preload();
        }
    }

    /**
     *  Check that the given symbol has a function ready.
     */
    bool symbol_ready(std::string symbol) {
        return funcs.count(symbol) > 0;
    }

    /**
     *  Construct a mapping of all symbols and from where they can be loaded.
     *  Populates compiler->libraries
     */
    size_t preload()
    {
        DIR *dir;
        struct dirent *ent;
        bool res = true;
        size_t nloaded = 0;

        //
        //  Index of a library containing multiple symbols
        //
        //  The file:
        //  BH_whateveryoumig_htlike.idx
        //
        //  Will create a new-line delimited list of symbol
        //  names which are available in:
        //
        //  BH_whateveryoumig_htlike.so
        //
        if ((dir = opendir (object_dir.c_str())) != NULL) {
            while ((ent = readdir (dir)) != NULL) {     // Go over dir-entries
                size_t fn_len = strlen(ent->d_name);
                if (fn_len<14) {
                    continue;
                }
                std::string filename(ent->d_name);
                std::string symbol;                     // BH_ADD_fff_CCC_3d
                std::string library;                    // BH_ADD_fff_CCC_3d_yAycwd

                if (0==filename.compare(fn_len-4, 4, ".idx")) {
                    // Library
                    library.assign(filename, 0, fn_len-4);

                    // Fill path to index filename
                    std::string index_fn = object_dir + "/" + filename;

                    std::ifstream symbol_file(index_fn.c_str());
                    for(std::string symbol; getline(symbol_file, symbol) && res;) {
                        if (0==libraries.count(symbol)) {
                            libraries.insert(
                                std::pair<std::string, std::string>(symbol, library)
                            );
                        }
                    }
                    symbol_file.close();
                }
            }
            closedir (dir);
        } else {
            throw std::runtime_error("Failed opening object-path.");
        }

        //
        //  A library containing a single symbol, the filename
        //  provides the symbol based on the naming convention:
        //
        //  BH_OPCODE_TYPESIG_LAYOUT_NDIM_XXXXXX.so
        //
        if ((dir = opendir (object_dir.c_str())) != NULL) {
            while((ent = readdir(dir)) != NULL) {
                size_t fn_len = strlen(ent->d_name);
                if (fn_len<14) {
                    continue;
                }
                std::string filename(ent->d_name);
                std::string symbol;                     // BH_ADD_fff_CCC_3d
                std::string library;                    // BH_ADD_fff_CCC_3d_yAycwd

                if ((0==filename.compare(0,3, "BH_")) && \
                    (0==filename.compare(fn_len-3, 3, ".so"))) {
                    symbol.assign(filename, 0, fn_len-10);  // Remove "_xxxxxx.so"
                    library.assign(filename, 0, fn_len-3);  // Remove ".so"

                    if (0==libraries.count(symbol)) {
                        libraries.insert(
                            std::pair<std::string, std::string>(symbol, library)
                        );
                    }
                }
            }
            closedir (dir);
        } else {
            throw std::runtime_error("Failed opening object-path.");
        }

        //cout << "PRELOADING... " << endl;
        //
        // This is the part that actually loads them...
        // This could be postponed...
        std::map<std::string, std::string>::iterator it;    // Iterator
        for(it=libraries.begin(); (it != libraries.end()) && res; ++it) {

            res = load(it->first, it->second);
            nloaded += res;
        }
        return nloaded;
    }

    /**
     *  Load a single symbol from library symbol into func-storage.
     */
    bool load(std::string symbol) {
        return load(symbol, libraries[symbol]);
    }
    bool load(std::string symbol, std::string library)
    {
        //cout << "LOAD: {" << symbol << ", " << library << "}" << endl;
        char *error_msg = NULL;             // Buffer for dlopen errors
        int errnum = 0;
        std::string library_path = object_dir + "/" + library + ".so";

        if (0==handles.count(library)) {    // Open library
            handles[library] = dlopen(
                library_path.c_str(),
                RTLD_NOW
            );
            errnum = errno;
        }
        if (!handles[library]) {            // Check that it opened
            error(
                errnum,
                "Failed openening library; dlopen(filename='%s', RTLF_NOW) failed.",
                library_path.c_str()
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
                library_path.c_str(),
                symbol.c_str()
            );
            //free(error_msg); TODO: This should not be freed!?
            return false;
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
        const char *mode = "w";
        int err;
        std::string kernel_path = kernel_dir +"/"+ symbol +"_"+ std::string(get_uid()) + ".c";
        kernel_fd = open(kernel_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
        if ((!kernel_fd) || (kernel_fd<1)) {
            err = errno;
            error(err, "Failed opening kernel-file [%s] in src_to_file(...).\n", kernel_path.c_str());
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

    /**
     *  Compile a shared library for the given symbol.
     *  The library name is constructed using the uid of the process.
     */
    bool compile(std::string symbol, const char* sourcecode, size_t source_len)
    {
        std::string library = symbol + "_" + std::string(get_uid());
        return compile(symbol, library, sourcecode, source_len);
    }

    /**
     *  Compile a shared library for the given symbol.
     */
    bool compile(std::string symbol, std::string library, const char* sourcecode, size_t source_len)
    {
        //
        // Constuct the compiler command
        std::string cmd = process_str +" "+ object_path +"/"+ library +".so";

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

        //
        // Update the library mapping such that a load for the symbol
        // can the resolve the library that it needs
        libraries.insert(
            std::pair<std::string, std::string>(symbol, library)
        );

        return true;
    }

    ~Compiler()
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

private:
    handle_storage handles;
    symbol_library_map libraries;
    char uid[7];
    std::string process_str;
    std::string object_dir;
    std::string kernel_dir;

};

#endif

