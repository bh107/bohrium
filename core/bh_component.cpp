/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#include <bh.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <boost/algorithm/string.hpp>

#ifdef _WIN32

#include <windows.h>
#include <dlfcn-win32.h>

#define HOME_INI_PATH "%APPDATA%\\bohrium\\config.ini"
#define SYSTEM_INI_PATH_1 "%PROGRAMFILES%\\bohrium\\config.ini"
#define SYSTEM_INI_PATH_2 "%PROGRAMFILES%\\bohrium\\config.ini"

//We need a buffer for path expansion
char _expand_buffer1[MAX_PATH];
char _expand_buffer2[MAX_PATH];
char _expand_buffer3[MAX_PATH];

//Nasty function renaming
#define snprintf _snprintf
#define strcasecmp _stricmp

#else

#include <dlfcn.h>
#include <limits.h>

#define HOME_INI_PATH "~/.bohrium/config.ini"
#define SYSTEM_INI_PATH_1 "/usr/local/etc/bohrium/config.ini"
#define SYSTEM_INI_PATH_2 "/usr/etc/bohrium/config.ini"

//We need a buffer for path expansion
char _expand_buffer[PATH_MAX];

#endif

// Check whether the given component exists.
static int component_exists(dictionary *dict, const char *name)
{
    char tmp[BH_COMPONENT_NAME_SIZE];
    snprintf(tmp, BH_COMPONENT_NAME_SIZE, "%s:type", name);
    char *s = iniparser_getstring(dict, tmp, NULL);
    return (NULL != s);
}

//Return the component type of the component named 'name'
static bh_component_type get_type(dictionary *dict, const char *name)
{
    char tmp[BH_COMPONENT_NAME_SIZE];
    snprintf(tmp, BH_COMPONENT_NAME_SIZE, "%s:type", name);
    char *s = iniparser_getstring(dict, tmp, NULL);
    if(s == NULL)
    {
        fprintf(stderr,"In section \"%s\" type is not set. "\
                       "Should be bridge, filter, fuser, vem or ve.\n",name);
        return BH_COMPONENT_ERROR;
    }
    else
    {
        if(!strcasecmp(s, "bridge"))
            return BH_BRIDGE;
        if(!strcasecmp(s, "vem"))
            return BH_VEM;
        if(!strcasecmp(s, "ve"))
            return BH_VE;
        if(!strcasecmp(s, "filter"))
            return BH_FILTER;
        if(!strcasecmp(s, "fuser"))
            return BH_FUSER;
        if(!strcasecmp(s, "stack"))
            return BH_STACK;
    }
    fprintf(stderr,"In section \"%s\" type is unknown: \"%s\" \n", name, s);
    return BH_COMPONENT_ERROR;
}

/**
 *  Extract component symbol from the given string.
 *  
 *  Expects a string like: "anythinglibbh_componenttype_COMPONENTNAME.anything"
 *
 *  No error-checking in this thing...
 *
 */
static void extract_symbol_from_path(const char* path, char* symbol)
{
    const char* prefix = "libbh_";                  // Component prefix
    int prefix_len = strlen(prefix);

    const char* filename = strstr(path, prefix);    // Filename
    filename += prefix_len;                         // Skip the prefix

    const char* sep = "_";
    int sep_len = strlen(sep);
    const char* name = strstr(filename, sep);
    name += sep_len;                                // Skip the seperator
    int name_len = strlen(name);

    strncpy(symbol, name, name_len);                // Copy the symbol
                                                            
    for(int i=0;                                    // Terminate at extension
        (symbol[i]!='\0') || (i>=(name_len-1));
        ++i) { 
        if (symbol[i] == '.') {                              
            *(symbol+i) = '\0';
            break;
        }
    }
}

static void *get_dlsym(void *handle, const char *name,
                       bh_component_type type, const char *fun)
{
    char tmp[1024];
    const char *stype;
    void *ret;
    if(type == BH_BRIDGE)
        stype = "bridge";
    else if(type == BH_VEM)
        stype = "vem";
    else if(type == BH_VE)
        stype = "ve";
    else if(type == BH_FILTER)
        stype = "filter";
    else if(type == BH_FUSER)
        stype = "fuser";
    else
    {
        fprintf(stderr, "Internal error get_dlsym() got unknown type\n");
        return NULL;
    }

    snprintf(tmp, BH_COMPONENT_NAME_SIZE, "bh_%s_%s_%s", stype, name, fun);
    dlerror();//Clear old errors.
    ret = dlsym(handle, tmp);
    char *err = dlerror();
    if(err != NULL)
    {
        fprintf(stderr, "Failed to load %s() from %s (%s).\n"
                        "Make sure to define all four interface functions, eg. the NODE-VEM "
                        "must define: bh_vem_node_init(), bh_vem_node_shutdown(), "
                        "bh_vem_node_extmethod(), and bh_vem_node_execute().\n",
                        fun, name, err);
        return NULL;
    }
    return ret;
}

/**
 *  Dynamically load component interface.
 *
 */
static bh_error component_dl_iface(dictionary* config, bh_component_iface* comp)
{
    if (!iniparser_find_entry(config, comp->name)) {        // Check for config-entry
        fprintf(stderr,
                "component_dl_iface: Failed retrieving config section for '%s'.\n",
                comp->name);
        return BH_ERROR;
    }

    char impl_inikey[BH_COMPONENT_NAME_SIZE+5];             // Get path to shared-object
    snprintf(impl_inikey, BH_COMPONENT_NAME_SIZE+5, "%s:impl", comp->name); 
    char *impl = iniparser_getstring(config, impl_inikey, NULL);
    if (impl == NULL) {
        fprintf(stderr,
                "component_dl_iface: Failed retrieving 'impl' for component '%s'\n",
                comp->name);
        return BH_ERROR;
    }

    bh_component_type comp_type = get_type(config, comp->name);
    if (comp_type == BH_COMPONENT_ERROR) {
        fprintf(stderr,
                "component_dl_iface: Failed getting type of component(%s).\n",
                comp->name);
    }

    char symbol[BH_COMPONENT_NAME_SIZE];
    extract_symbol_from_path(impl, symbol);                 // Get the component "symbol"

    //
    // Load component-interface functions
    //

    void *lib_handle = dlopen(impl, RTLD_NOW);              // Open the library
    if (lib_handle == NULL) {
        fprintf(stderr,
                "component_dl_iface: Error in [%s:impl]: %s\n",
                comp->name,
                dlerror());
        return BH_ERROR;
    }
    comp->lib_handle = lib_handle;                          // Store library handle

    // Grab component interface symbols init, shutdown, execute, extmethod
    comp->init = (bh_init)get_dlsym(lib_handle, symbol, comp_type, "init");
    if (comp->init == NULL) {
        fprintf(stderr,
                "component_dl_iface: Failed retrieving iface-init for %s.",
                comp->name);
        return BH_ERROR;
    }
    comp->shutdown = (bh_shutdown)get_dlsym(lib_handle, symbol, comp_type, "shutdown");
    if (comp->shutdown == NULL) {
        fprintf(stderr,
                "component_dl_iface: Failed retrieving iface-shutdown for %s.",
                comp->name);
        return BH_ERROR;
    }
    comp->execute = (bh_execute)get_dlsym(lib_handle, symbol, comp_type, "execute");
    if (comp->execute == NULL) {
        fprintf(stderr,
                "component_dl_iface: Failed retrieving iface-execute for %s.",
                comp->name);
        return BH_ERROR;
    }
    comp->extmethod = (bh_extmethod)get_dlsym(lib_handle, symbol, comp_type, "extmethod");
    if (comp->extmethod == NULL) {
        fprintf(stderr,
                "component_dl_iface: Failed retrieving iface-extmethod for %s.",
                comp->name);
        return BH_ERROR;
    }
    return BH_SUCCESS;
}

/* Initilize children of the given component
 *
 * @self   The component of which children will be initialized
 * @stack  The stack configuration to use, if NULL use children-chaining.
 * @return Error codes (BH_SUCCESS, BH_ERROR)
 */
static bh_error component_children_init(bh_component *self, char* stack)
{
    self->nchildren = 0;

    char child_inikey[BH_COMPONENT_NAME_SIZE];  // Where to look for children
    if (stack) {
        snprintf(child_inikey, BH_COMPONENT_NAME_SIZE, "%s:%s", stack, self->name);
    } else {
        snprintf(child_inikey, BH_COMPONENT_NAME_SIZE, "%s:children", self->name);
    }

    char *children_str = iniparser_getstring(self->config, child_inikey, NULL);
    if (children_str == NULL) {                 // No children -- we are finished
        return BH_SUCCESS;
    }

    char *child_name = strtok(children_str, ",");
    while(child_name != NULL) {
        bh_component_iface *child = &self->children[self->nchildren];   // Grab child
        strncpy(child->name, child_name, BH_COMPONENT_NAME_SIZE);       // Store name
        component_dl_iface(self->config, child);                        // Load interface

        ++(self->nchildren);                                            // Increment count
        if (self->nchildren > BH_COMPONENT_MAX_CHILDS) {
            fprintf(stderr,
                    "Number of children of %s is greater "
                    "than BH_COMPONENT_MAX_CHILDS.\n", self->name);
            return BH_ERROR;
        }

        child_name = strtok(NULL, ",");                                 // Go to next child
    }
    return BH_SUCCESS;
}

/** 
 * Find configuration file
 *
 */
bh_error bh_component_config_find(bh_component *self)
{
    const char* homepath = HOME_INI_PATH;
    const char* syspath1 = SYSTEM_INI_PATH_1;
    const char* syspath2 = SYSTEM_INI_PATH_2;
   
    //
    // Find the configuration file
    //

    // Start by looking a path set via environment variable.
    const char *env = getenv("BH_CONFIG");
    if (env != NULL)
    {
        FILE *fp = fopen(env,"r");
        if( fp )
            fclose(fp);
        else
            env = NULL;//Did not exist.
    }

    // Then the home directory.
    if(env == NULL)
    {
#if _WIN32
        DWORD result = ExpandEnvironmentStrings(
            homepath,
            _expand_buffer1,
            MAX_PATH-1
        );

        if (result != 0)
        {
            homepath = _expand_buffer1;
        }
#else
        char* h = getenv("HOME");
        if (h != NULL)
        {
            snprintf(_expand_buffer, PATH_MAX, "%s/%s", h, homepath+1);
            homepath = _expand_buffer;
        }
#endif
        FILE *fp = fopen(homepath,"r");
        if( fp ) {
            env = homepath;
            fclose(fp);
        }
    }

    //And then system-wide.
    if(env == NULL)
    {
#if _WIN32
        DWORD result = ExpandEnvironmentStrings(
            syspath1,
            _expand_buffer2,
            MAX_PATH-1
        );

        if(result != 0)
        {
            syspath1 = _expand_buffer2;
        }
#endif
        FILE *fp = fopen(syspath1,"r");
        if(fp)
        {
            env = syspath1;
            fclose(fp);
        }
    }

    //And then system-wide.
    if(env == NULL)
    {
#if _WIN32
        DWORD result = ExpandEnvironmentStrings(
            syspath2,
            _expand_buffer3,
            MAX_PATH-1
        );

        if(result != 0)
        {
            syspath2 = _expand_buffer3;
        }
#endif
        FILE *fp = fopen(syspath2,"r");
        if(fp)
        {
            env = syspath2;
            fclose(fp);
        }
    }
    // We could not find the configuration file anywhere
    if(env == NULL)
    {
        fprintf(stderr, "Error: Bohrium could not find the config file.\n"
            " The search is:\n"
            "\t* The environment variable BH_CONFIG.\n"
            "\t* The home directory \"%s\".\n"
            "\t* The local directory \"%s\".\n"
            "\t* And system-wide \"%s\".\n", homepath, syspath1, syspath2);
        return BH_ERROR;
    }

    // Load the bohrium configuration file
    self->config = iniparser_load(env);
    if(self->config == NULL)
    {
        fprintf(stderr, "Error: Bohrium could not read the config file.\n");
        return BH_ERROR;
    }
    return BH_SUCCESS;
}

/* Initilize the component object
 *
 * @self   The component object to initilize
 * @name   The name of the component. If NULL "bridge" will be used.
 * @return Error codes (BH_SUCCESS, BH_ERROR)
 */
bh_error bh_component_init(bh_component *self, const char* name)
{
    memset(self, 0, sizeof(bh_component));  // Clear component-memory
                                            // Find configuration-file
    bh_error found_config = bh_component_config_find(self);
    if (BH_SUCCESS != found_config) {
        return found_config;
    }

    char* stack = getenv("BH_STACK");       // Get the stack from environment
    int default_stack = 0;
    if (NULL == stack) {                    // Default to "stack_default"
        stack = (char*)"stack_default";
        default_stack = 1;
    }
    
    int stack_exists = component_exists(self->config, stack);
    if ((!default_stack) && (!stack_exists)) {
        fprintf(stderr, "The requested stack configuration(%s) does not exist,"
                        " falling back to children-chaining.\n", stack);
    }
    if (!stack_exists) {
        stack = NULL;
    }

    if (name == NULL) {                                 // Store component name
        if (stack) {
            strcpy(self->name, stack);
        } else {
            strcpy(self->name, "bridge");
        }
    } else {
        strcpy(self->name, name);
    }
    
    self->type = get_type(self->config, self->name);    // Store type
    if (BH_COMPONENT_ERROR == self->type) {
        return BH_ERROR;
    }

    return component_children_init(self, stack);        // Initialize children
}

/* Destroyes the component object.
 *
 * @self   The component object to destroy
 */
void bh_component_destroy(bh_component *self)
{
    iniparser_freedict(self->config);
}

/* Retrieves an extension method implementation.
 *
 * @self      The component object.
 * @name      Name of the extension method e.g. matmul
 * @extmethod Pointer to the method (output)
 * @return    Error codes (BH_SUCCESS, BH_ERROR, BH_OUT_OF_MEMORY,
 *                         BH_EXTMETHOD_NOT_SUPPORTED)
 */
bh_error bh_component_extmethod(const bh_component *self,
                                const char *name,
                                bh_extmethod_impl *extmethod)
{
    //We search the libs in the config file to find the user-defined function.
    char *lib_paths = bh_component_config_lookup(self,"libs");
    if(lib_paths != NULL)
    {
        //Lets make a working copy
        lib_paths = strdup(lib_paths);
        if(lib_paths == NULL)
            return BH_OUT_OF_MEMORY;

        //Handle one library path at a time.
        char *path = strtok(lib_paths,",");
        while(path != NULL)
        {
            void *lib_handle = dlopen(path, RTLD_NOW);
            if(lib_handle != NULL)
            {
                char tname[BH_COMPONENT_NAME_SIZE];
                sprintf(tname, "bh_%s", name);
                dlerror();//Clear old errors.
                *extmethod = (bh_extmethod_impl)dlsym(lib_handle, tname);
                if(dlerror() == NULL)//No errors, we found the function
                {
                    free(lib_paths);
                    return BH_SUCCESS;
                }
            }
            path = strtok(NULL,",");
        }
        free(lib_paths);
    }
    *extmethod = NULL;//Make sure it is NULL on error.
    return BH_EXTMETHOD_NOT_SUPPORTED;
}

/* Look up a key in the environment
 * Private function
 *
 * @component The component.
 * @key       The key to lookup in the config file
 * @return    The value if found, otherwise NULL
 */
static char *lookup_env(const bh_component *component, const char* key)
{
    using namespace std;
    stringstream ss;
    ss << "BH_" << component->name << "_" << key;
    string s = boost::to_upper_copy(ss.str());
    return getenv(s.c_str());
}

/* Look up a key in the config file
 *
 * @component The component.
 * @key       The key to lookup in the config file
 * @return    The value if found, otherwise NULL
 */
char* bh_component_config_lookup(const bh_component *component, const char* key)
{
    char *env = lookup_env(component, key);
    if(env != NULL)
        return env;

    char dictkey[BH_COMPONENT_NAME_SIZE];
    snprintf(dictkey, BH_COMPONENT_NAME_SIZE, "%s:%s", component->name, key);
    return iniparser_getstring(component->config, dictkey, NULL);
}

/*
 * @brief     Lookup a keys value in the config fil converted to a bool
 * @component The component.
 * @key       The key to lookup in the config file
 * @notfound  Value to return in case of error
 * @return    bool
*/
bool bh_component_config_lookup_bool(const bh_component *component,
                                     const char* key, bool notfound)
{
    char* val ;
    bool ret ;
    val = bh_component_config_lookup(component, key);
    if (val == NULL)
        return notfound ;
    if (val[0]=='y' || val[0]=='Y' || val[0]=='1' || val[0]=='t' || val[0]=='T') {
        ret = true;
    } else if (val[0]=='n' || val[0]=='N' || val[0]=='0' || val[0]=='f' || val[0]=='F') {
        ret = false;
    } else {
        ret = notfound ;
    }
    return ret;
}

/*
 * @brief     Lookup a keys value in the config fil converted to an int
 * @component The component.
 * @key       The key to lookup in the config file
 * @notfound  Value to return in case of error
 * @return    int
*/
int bh_component_config_lookup_int(const bh_component *component,
                                   const char* key, int notfound)
{
    char* val;
    val = bh_component_config_lookup(component, key);
    if (val == NULL)
        return notfound;
    return (int)strtol(val, NULL, 0);
}

/*
 * @brief     Lookup a keys value in the config fil converted to a double
 * @component The component.
 * @key       The key to lookup in the config file
 * @notfound  Value to return in case of error
 * @return    double
*/
double bh_component_config_lookup_double(const bh_component *component,
                                         const char* key, double notfound)
{
    char* val;
    val = bh_component_config_lookup(component, key);
    if (val == NULL)
        return notfound;
    return atof(val);
}

bh_error bh_component_config_int_option(const bh_component* component,
                                        const char* option_name,
                                        int min,
                                        int max,
                                        bh_intp* option)
{
    char* raw = bh_component_config_lookup(component, option_name);
    if (!raw) {
        fprintf(stderr, "parameter(%s) is missing.\n", option_name);
        return BH_ERROR;
    }
    *option = (bh_intp)atoi(raw);
    if ((*option < min) || (*option > max)) {
        fprintf(
            stderr,
            "%s should be within range [%d,%d].\n",
            option_name, min, max
        );
        return BH_ERROR;
    }
    return BH_SUCCESS;
}

bh_error bh_component_config_string_option(const bh_component* component,
                                           const char* option_name,
                                           char** option)
{
    *option = bh_component_config_lookup(component, option_name);
    if (!option) {
        fprintf(stderr, "%s is missing.\n", option_name);
        return BH_ERROR;
    }
    return BH_SUCCESS;
}

bh_error bh_component_config_path_option(const bh_component* component,
                                         const char* option_name,
                                         char** option)
{
    *option = bh_component_config_lookup(component, option_name);

    if (!option) {
        fprintf(stderr, "Path is not set; option (%s).\n", option_name);
        return BH_ERROR;
    }
    if (0 != access(*option, F_OK)) {
        if (ENOENT == errno) {
            fprintf(stderr, "Path does not exist; path (%s).\n", *option);
        } else if (ENOTDIR == errno) {
            fprintf(stderr, "Path is not a directory; path (%s).\n", *option);
        } else {
            fprintf(stderr, "Path is broken somehow; path (%s).\n", *option);
        }
        return BH_ERROR;
    }
    return BH_SUCCESS;
}
