#include "store.hpp"

using namespace std;
namespace bohrium {
namespace engine {
namespace cpu {

Store::Store(const string object_dir) : object_dir(object_dir)
{
    char uid[7];    // Create an identifier with low collision...    
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
    DEBUG("++ Store::~Store()");
    DEBUG("-- Store::~Store()");
}

string Store::text(void)
{
    stringstream ss;
    ss << "Store(\"" << object_dir << "\") : uid(" << this->uid << ");" << endl;
    return ss.str();
}

/**
 *  Get the id of the store.
 */
string Store::get_uid(void)
{
    DEBUG("   Store::get_uid(void) : uid(" << this->uid << ");");
    return this->uid;
}

/**
 *  Check that the given symbol has an object ready.
 */
bool Store::symbol_ready(string symbol)
{
    DEBUG("   Store::symbol_ready("<< symbol << ") : return(" << (funcs.count(symbol) > 0) << ");");
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

void Store::add_symbol(string symbol, string library)
{
    libraries.insert(pair<string, string>(symbol, library));
}

/**
 *  Load a single symbol from library symbol into func-storage.
 */
bool Store::load(string symbol) {
    DEBUG("   Store::load("<< symbol << ");");

    return load(symbol, libraries[symbol]);
}

bool Store::load(string symbol, string library)
{
    DEBUG("   Store::load("<< symbol << ", " << library << ");");
    
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
