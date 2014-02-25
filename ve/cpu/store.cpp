#include "store.hpp"

using namespace std;
namespace bohrium {
namespace engine {
namespace cpu {

Store::Store(string object_dir) : object_dir(object_dir)
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

/**
 *  Get the id of the store.
 */
const char* Store::get_uid(void)
{
    return uid;
}

/**
 *  Check that the given symbol has an object ready.
 */
bool Store::symbol_ready(string symbol)
{
    return funcs.count(symbol) > 0;
}

/**
 *  Construct a mapping of all symbols and from where they can be loaded.
 *  Populates compiler->libraries
 */
size_t Store::preload()
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
            string filename(ent->d_name);
            string symbol;                     // BH_ADD_fff_CCC_3d
            string library;                    // BH_ADD_fff_CCC_3d_yAycwd

            if (0==filename.compare(fn_len-4, 4, ".idx")) {
                // Library
                library.assign(filename, 0, fn_len-4);
               
                // Fill path to index filename 
                string index_fn = object_dir + "/" + filename;

                ifstream symbol_file(index_fn);
                for(string symbol; getline(symbol_file, symbol) && res;) {
                    if (0==libraries.count(symbol)) {
                        libraries.insert(
                            pair<string, string>(symbol, library)
                        );
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
    //  BH_OPCODE_TYPESIG_LAYOUT_NDIM_XXXXXX.so
    //
    if ((dir = opendir (object_dir.c_str())) != NULL) {
        while((ent = readdir(dir)) != NULL) {
            size_t fn_len = strlen(ent->d_name);
            if (fn_len<14) {
                continue;
            }
            string filename(ent->d_name);
            string symbol;                     // BH_ADD_fff_CCC_3d
            string library;                    // BH_ADD_fff_CCC_3d_yAycwd

            if ((0==filename.compare(0,3, "BH_")) && \
                (0==filename.compare(fn_len-3, 3, ".so"))) { 
                symbol.assign(filename, 0, fn_len-10);  // Remove "_xxxxxx.so"
                library.assign(filename, 0, fn_len-3);  // Remove ".so"

                if (0==libraries.count(symbol)) {
                    libraries.insert(
                        pair<string, string>(symbol, library)
                    );
                }
            }
        }
        closedir (dir);
    } else {
        throw runtime_error("Failed opening object-path.");
    }

    //cout << "PRELOADING... " << endl;
    //
    // This is the part that actually loads them...
    // This could be postponed...
    map<string, string>::iterator it;    // Iterator
    for(it=libraries.begin(); (it != libraries.end()) && res; ++it) {

        res = load(it->first, it->second);
        nloaded += res;
    }
    return nloaded;
}

/**
 *  Load a single symbol from library symbol into func-storage.
 */
bool Store::load(string symbol) {
    return load(symbol, libraries[symbol]);
}

bool Store::load(string symbol, string library)
{
    //cout << "LOAD: {" << symbol << ", " << library << "}" << endl;
    char *error_msg = NULL;             // Buffer for dlopen errors
    int errnum = 0;
    string library_path = object_dir + "/" + library + ".so";

    if (0==handles.count(library)) {    // Open library
        handles[library] = dlopen(
            library_path.c_str(),
            RTLD_NOW
        );
        errnum = errno;
    }
    if (!handles[library]) {            // Check that it opened
        utils::error(
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
        utils::error(
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

}}}