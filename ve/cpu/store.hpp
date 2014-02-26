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
#include <fcntl.h>
#include <errno.h>

#include "tac.h"
#include "utils.hpp"

namespace bohrium {
namespace engine {
namespace cpu {

typedef void (*func)(operand_t args[]);

//
//  Retrieve a function pointer for the symbol (SYMBOL -> func)
typedef std::map<std::string, func> func_storage;

//
//  Retrieve the library handle for a library (LIBRARY -> handle)
typedef std::map<std::string, void*> handle_storage;

//
//  In which library is the symbol stored (SYMBOL -> LIBRARY)
typedef std::map<std::string, std::string> symbol_library_map;

class Store {
public:
    Store(const std::string object_dir);
    std::string text();

    bool symbol_ready(std::string symbol);
    std::string get_uid(void);
    bool load(std::string symbol);
    bool load(std::string symbol, std::string library);
    size_t preload();

    func_storage funcs;

private:
    handle_storage handles;
    symbol_library_map libraries;
    std::string object_dir;
    std::string uid;
};

}}}
#endif