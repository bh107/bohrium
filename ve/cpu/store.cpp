#include "store.hpp"

#include "thirdparty/MurmurHash3.h"

using namespace std;
namespace bohrium {
namespace engine {
namespace cpu {

const char Store::TAG[] = "Store";

Store::Store(const string object_directory, const string kernel_directory) 
: object_directory(object_directory), kernel_directory(kernel_directory), kernel_prefix("KRN_"), library_prefix("LIB_")
{
    //
    // Create an identifier with low collision...
    char uid[7];    
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    srand(getpid());
    for (int i = 0; i < 7; ++i) {
        uid[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    uid[6] = 0;

    this->uid = string(uid);

    //cout << "Storage " << this->uid << endl;
}

Store::~Store()
{
    for(handle_storage::iterator it=handles.begin(); it!=handles.end(); ++it) {
        dlclose(it->second);
    }
}

string Store::text(void)
{
    stringstream ss;
    ss << "Store(\"" << object_directory << "\") : uid(" << this->uid << ");" << endl;
    return ss.str();
}

/**
 *  Get the id of the store.
 */
string Store::get_uid(void)
{
    return this->uid;
}

string Store::obj_filename(const string symbol)
{
    return  this->kernel_prefix     +\
            symbol                  +\
            "_"                     +\
            this->uid               +\
            ".so";
}

string Store::obj_abspath(const string symbol)
{
    return  this->object_directory  +\
            "/"                     +\
            this->obj_filename(symbol);
}

string Store::src_filename(const string symbol)
{
    return  this->kernel_prefix     +\
            symbol                  +\
            "_"                     +\
            this->uid               +\
            ".c";
}

string Store::src_abspath(const string symbol)
{
    return  this->kernel_directory  +\
            "/"                     +\
            this->src_filename(symbol);
}

/**
 *  Check that the given symbol has an object ready.
 */
bool Store::symbol_ready(const string symbol)
{
    return funcs.count(symbol) > 0;
}

/**
 *  Construct a mapping of all symbols and from where they can be loaded.
 *  Populates compiler->libraries
 */
size_t Store::preload(void)
{
    DIR *dir;
    struct dirent *ent;
    bool res = true;
    size_t nloaded = 0;

    //
    //  Index of a library containing multiple symbols
    //
    //  The file:
    //  LIB_[a-z0-9]+_xxxxxx.idx
    //
    //  Contains a new-line delimited list of names such as:
    //
    //  KRN_\d+_xxxxxx
    //
    //  Which can be loaded from:
    //
    //  LIB_[a-z0-9]+_xxxxxx.so
    //
    if ((dir = opendir(object_directory.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {     // Go over dir-entries
            size_t fn_len = strlen(ent->d_name);
            if (fn_len<14) {
                continue;
            }
            string filename(ent->d_name);
            string symbol;                     // KRN_\d+
            string library;                    // LIB_whatever_xxxxxx.so

            if (0==filename.compare(fn_len-4, 4, ".idx")) {
                string basename;
                // Remove the ".idx" extension
                basename.assign(filename, 0, filename.length()-4);

                // Add the ".so" extension
                library = basename +".so";
               
                //
                // Construct the absolute path to the file since we need
                // to open and read it.
                string index_fn = object_directory  +\
                                  "/"               +\
                                  filename;
                ifstream symbol_file(index_fn.c_str(), ifstream::in);
                for(string symbol; getline(symbol_file, symbol) && res;) {
                    if (0==libraries.count(symbol)) {
                        add_symbol(symbol, library);
                    }
                }
                symbol_file.close();
            }
        }
        closedir (dir);
    } else {
        throw runtime_error("Failed opening object-path.");
    }

    //
    //  A library containing a single symbol, the filename
    //  provides the symbol based on the naming convention:
    //
    //  KRN_[\d+]_XXXXXX.so
    //
    if ((dir = opendir (object_directory.c_str())) != NULL) {
        while((ent = readdir(dir)) != NULL) {
            size_t fn_len = strlen(ent->d_name);
            if (fn_len<14) {
                continue;
            }
            string filename(ent->d_name);
            string symbol;                     // BH_ADD_fff_CCC_3d
           
            // Must begin with "KRN_" 
            // Must end with ".so"
            if ((0==filename.compare(0, this->kernel_prefix.length(), this->kernel_prefix)) && \
                (0==filename.compare(fn_len-3, 3, ".so"))) {

                // Extract the symbol "KRN_(d+)_xxxxxx.so"
                symbol.assign(
                    filename,
                    this->kernel_prefix.length(),
                    fn_len-10-this->kernel_prefix.length()
                );

                if (0==libraries.count(symbol)) {
                    add_symbol(symbol, filename);
                }
            }
        }
        closedir (dir);
    } else {
        throw runtime_error("Failed opening object-path.");
    }

    //
    // This is the part that actually loads them...
    // This could be postponed...
    map<const string, const string>::iterator it;    // Iterator
    for(it=libraries.begin(); (it != libraries.end()) && res; ++it) {
        res = load(it->first, it->second);
        nloaded += res;
    }

    return nloaded;
}

void Store::add_symbol(const string symbol, const string library)
{
    libraries.insert(pair<string, string>(symbol, library));
}

/**
 *  Load a single symbol from library symbol into func-storage.
 */
bool Store::load(const string symbol)
{
    return load(symbol, libraries[symbol]);
}

bool Store::load(const string symbol, const string library)
{
    char *error_msg = NULL;             // Buffer for dlopen errors
    
    string library_abspath = this->object_directory+"/"+library;

    dlerror();                          // Clear any existing error then,
    if (0==handles.count(library)) {    // Open library
        void* lib_handle = dlopen(library_abspath.c_str(), RTLD_NOW);
        if (NULL==lib_handle) {
            error_msg = dlerror();
            fprintf(stderr, "dlopen(..., RTLD_NOW) failed, dlerror() msg: [%s]\n", error_msg);
            return false; 
        }
        handles[library] = lib_handle;
    }

    dlerror();                          // Clear any existing error then,
    funcs[symbol] = (func)dlsym(        // Load symbol/function
        handles[library],
        (kernel_prefix+symbol).c_str()
    );
    error_msg = dlerror();
    if (error_msg) {
        perror("dlsym()");
        fprintf(
            stderr,
            "dlsym( handle='%s', symbol='%s' )\n",
            library.c_str(),
            symbol.c_str()
        );
        return false;
    }
    return true;
}

}}}
