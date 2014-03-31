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
}

Store::~Store()
{
    DEBUG(TAG,"~Store()");
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
    DEBUG(TAG,"get_uid(void) : uid(" << this->uid << ");");
    return this->uid;
}

string Store::obj_filename(string symbol)
{
    return  this->kernel_prefix     +\
            symbol                  +\
            "_"                     +\
            this->uid               +\
            ".so";
}

string Store::obj_abspath(string symbol)
{
    return  this->object_directory  +\
            "/"                     +\
            this->obj_filename(symbol);
}

string Store::src_filename(string symbol)
{
    return  this->kernel_prefix     +\
            symbol                  +\
            "_"                     +\
            this->uid               +\
            ".c";
}

string Store::src_abspath(string symbol)
{
    return  this->kernel_directory  +\
            "/"                     +\
            this->src_filename(symbol);
}

/**
 *  Check that the given symbol has an object ready.
 */
bool Store::symbol_ready(string symbol)
{
    DEBUG(TAG,"symbol_ready("<< symbol << ") : return(" << (funcs.count(symbol) > 0) << ");");
    return funcs.count(symbol) > 0;
}

/**
 *  Construct a mapping of all symbols and from where they can be loaded.
 *  Populates compiler->libraries
 */
size_t Store::preload(void)
{
    DEBUG(TAG,"preload(void)");
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
        DEBUG(TAG,"preload(...) -- GOING MULTI!");
        while ((ent = readdir (dir)) != NULL) {     // Go over dir-entries
            size_t fn_len = strlen(ent->d_name);
            if (fn_len<14) {
                continue;
            }
            string filename(ent->d_name);
            DEBUG(TAG," We have a potential: " << filename);
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
                DEBUG(TAG," MULTI: " << library << " ||| " << index_fn);
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
        DEBUG(TAG," GOING SINGLE!");
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
    map<string, string>::iterator it;    // Iterator
    for(it=libraries.begin(); (it != libraries.end()) && res; ++it) {
        res = load(it->first, it->second);
        nloaded += res;
    }

    DEBUG(TAG," preload(void) : nloaded("<< nloaded << ");");

    return nloaded;
}

void Store::add_symbol(string symbol, string library)
{
    DEBUG(TAG,"add_symbol("<< symbol <<", "<< library <<");");
    libraries.insert(pair<string, string>(symbol, library));
}

/**
 *  Load a single symbol from library symbol into func-storage.
 */
bool Store::load(string symbol) {
    DEBUG(TAG,"load("<< symbol << ");");

    return load(symbol, libraries[symbol]);
}

bool Store::load(string symbol, string library)
{
    DEBUG(TAG,"load("<< symbol << ", " << library << ");");
    
    char *error_msg = NULL;             // Buffer for dlopen errors
    int errnum = 0;
    
    string library_abspath = this->object_directory+"/"+library;

    if (0==handles.count(library)) {    // Open library
        handles[library] = dlopen(
            library_abspath.c_str(),
            RTLD_NOW
        );
        errnum = errno;
    }
    if (!handles[library]) {            // Check that it opened
        utils::error(
            errnum,
            "Store::load(...,...) : dlopen(filename='%s', RTLF_NOW).",
            library_abspath.c_str()
        );
        return false;
    }

    dlerror();                          // Clear any existing error then,
    funcs[symbol] = (func)dlsym(        // Load symbol/function
        handles[library],
        (kernel_prefix+symbol).c_str()
    );
    error_msg = dlerror();
    if (error_msg) {
        utils::error(
            error_msg,
            "dlsym( handle='%s', symbol='%s' )\n",
            library.c_str(),
            symbol.c_str()
        );
        //free(error_msg); TODO: This should not be freed!?
        return false;
    }
    return true;
}

}}}
