#ifndef __BH_VE_CPU_STORE
#define __BH_VE_CPU_STORE

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
#include <dirent.h>
#include <dlfcn.h>
#include <unistd.h>
#include <errno.h>

#include "tac.h"
#include "utils.hpp"

namespace bohrium {
namespace engine {
namespace cpu {

typedef void (*func)(operand_t** args, iterspace_t* iterspace);

//
//  Retrieve a function pointer for the symbol (SYMBOL -> func)
typedef std::map<const std::string, func> func_storage;

//
//  Retrieve the library handle for a library (LIBRARY -> handle)
typedef std::map<const std::string, void*> handle_storage;

//
//  In which library is the symbol stored (SYMBOL -> LIBRARY)
typedef std::map<const std::string, const std::string> symbol_library_map;

class Store {
public:
    func_storage funcs;

    Store(const std::string object_directory, const std::string kernel_directory);
    ~Store();
    std::string text();

    void add_symbol(const std::string symbol, const std::string library);

    bool symbol_ready(const std::string symbol);
    bool load(const std::string symbol);
    bool load(const std::string symbol, const std::string library);
    size_t preload();

    std::string get_uid(void);

    std::string obj_filename(const std::string symbol);
    std::string src_filename(const std::string symbol);

    std::string obj_abspath(const std::string symbol);
    std::string src_abspath(const std::string symbol);

private:
    handle_storage handles;
    symbol_library_map libraries;
    std::string object_directory;
    std::string kernel_directory;
    std::string uid;
    std::string kernel_prefix;
    std::string library_prefix;

    static const char TAG[];
};

}}}
#endif
