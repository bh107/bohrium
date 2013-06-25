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
#include <iniparser.h>
#include <string.h>
#include <assert.h>

#ifdef _WIN32

#include <windows.h>
#include <dlfcn-win32.h>

#define HOME_INI_PATH "%APPDATA%\\bohrium\\config.ini"
#define SYSTEM_INI_PATH "%PROGRAMFILES%\\bohrium\\config.ini"

//We need a buffer for path expansion
char _expand_buffer1[MAX_PATH];
char _expand_buffer2[MAX_PATH];

//Nasty function renaming
#define snprintf _snprintf
#define strcasecmp _stricmp

#else

#include <dlfcn.h>
#include <limits.h>

#define HOME_INI_PATH "~/.bohrium/config.ini"
#define SYSTEM_INI_PATH "/etc/bohrium/config.ini"

//We need a buffer for path expansion
char _expand_buffer[PATH_MAX];

#endif


static bh_component_type get_type(dictionary *dict, const char *name)
{
    char tmp[BH_COMPONENT_NAME_SIZE];
    snprintf(tmp, BH_COMPONENT_NAME_SIZE, "%s:type", name);
    char *s = iniparser_getstring(dict, tmp, NULL);
    if(s == NULL)
    {
        fprintf(stderr,"In section \"%s\" type is not set. "\
                       "Should be bridge, vem or ve.\n",name);
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
    }
    fprintf(stderr,"In section \"%s\" type is unknown: \"%s\" \n",
            name, s);
    return BH_COMPONENT_ERROR;
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
    else
    {
        fprintf(stderr, "get_dlsym - unknown component type.\n");
        return NULL;
    }

    snprintf(tmp, BH_COMPONENT_NAME_SIZE, "bh_%s_%s_%s", stype, name, fun);
    dlerror();//Clear old errors.
    ret = dlsym(handle, tmp);
    char *err = dlerror();
    if(err != NULL)
    {
        fprintf(stderr, "[%s:type]%s\n", name, err);
        return NULL;
    }
    return ret;
}

/* Setup the root component, which normally is the bridge.
 *
 * @name The name of the root component. If NULL "bridge"
         will be used.
 * @return The root component in the configuration.
 */
bh_component *bh_component_setup(const char* component_name)
{
    const char* homepath = HOME_INI_PATH;
    const char* syspath = SYSTEM_INI_PATH;
    const char *name;
    if(component_name == NULL)
        name = "bridge";
    else
        name = component_name;

    bh_component *com = (bh_component*)malloc(sizeof(bh_component));
    const char *env;
    if(com == NULL)
    {
        fprintf(stderr, "bh_component_setup(): out of memory.\n");
        return NULL;
    }

    //Clear memory so we do not have any random pointers
    memset(com, 0, sizeof(bh_component));

    if(name == NULL)
        strcpy(com->name, "bridge"); //The default config root keyword.
    else
        strcpy(com->name, name);

    //The environment variable has precedence.
    env = getenv("BH_CONFIG");
    if (env != NULL)
    {
        FILE *fp = fopen(env,"r");
        if( fp )
            fclose(fp);
        else
            env = NULL;//Did not exist.
    }

    //Then the home directory.
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

    //And finally system-wide.
    if(env == NULL)
    {
#if _WIN32
        DWORD result = ExpandEnvironmentStrings(
            syspath,
            _expand_buffer2,
            MAX_PATH-1
        );

        if (result != 0)
        {
            syspath = _expand_buffer2;
        }
#endif

        FILE *fp = fopen(syspath,"r");
        if( fp ) {
            env = syspath;
            fclose(fp);
        }
    }

    if(env == NULL)
    {
        fprintf(stderr, "Error: Bohrium could not find the config file.\n"
            " The search is:\n"
            "\t* The environment variable BH_CONFIG.\n"
            "\t* The home directory \"%s\".\n"
            "\t* And system-wide \"%s\".\n", homepath, syspath);
        free(com);
        return NULL;
    }

    com->config = iniparser_load(env);
    if(com->config == NULL)
    {
        fprintf(stderr, "Error: Bohrium could not read the config file.\n");
        free(com);
        return NULL;
    }

    com->type = get_type(com->config, com->name);

    if(strcmp("bridge", name) != 0)//This is not the bridge
    {
        char tmp[BH_COMPONENT_NAME_SIZE];
        snprintf(tmp, BH_COMPONENT_NAME_SIZE, "%s:impl",name);
        char *impl = iniparser_getstring(com->config, tmp, NULL);
        if(impl == NULL)
        {
            fprintf(stderr,"In section \"%s\" impl is not set.\n",name);
            return NULL;
        }
        com->lib_handle = dlopen(impl, RTLD_NOW);
        if(com->lib_handle == NULL)
        {
            fprintf(stderr, "Error in [%s:impl]: %s\n", name, dlerror());
            return NULL;
        }
    }
    else
        com->lib_handle = NULL;//The bridge do not have a .so file
    return com;
}

/* Retrieves the children components of the parent.
 *
 * @parent The parent component (input).
 * @count Number of children components(output).
 * @children Array of children components (output).
 * @return Error code (BH_SUCCESS).
 */
bh_error bh_component_children(bh_component *parent, bh_intp *count,
                                     bh_component **children[])
{
    char tmp[BH_COMPONENT_NAME_SIZE];
    bh_error result;
    char *child;
    size_t c;
    *count = 0;
    snprintf(tmp, BH_COMPONENT_NAME_SIZE, "%s:children",parent->name);
    char *tchildren = iniparser_getstring(parent->config, tmp, NULL);
    if(tchildren == NULL)
    {
        fprintf(stderr, "bh_component_setup(): children missing from config.\n");
		return BH_ERROR;
	}

    *children = (bh_component**)malloc(BH_COMPONENT_MAX_CHILDS * sizeof(bh_component *));
    if(*children == NULL)
    {
        fprintf(stderr, "bh_component_setup(): out of memory.\n");
        return BH_OUT_OF_MEMORY;
    }
    //Since we do not use all the data here, it is good for debugging if the rest is null pointers
    memset(*children, 0, BH_COMPONENT_MAX_CHILDS * sizeof(bh_component *));

	//Assume all goes well
	result = BH_SUCCESS;

    //Handle one child at a time.
    child = strtok(tchildren,",");
    while(child != NULL && *count < BH_COMPONENT_MAX_CHILDS)
    {
        (*children)[*count] = (bh_component*)malloc(sizeof(bh_component));
        bh_component *com = (*children)[*count];

        //Save component name.
        strncpy(com->name, child, BH_COMPONENT_NAME_SIZE);
        //Save configuration dictionary.
        com->config = parent->config;
        //Save component type.
        com->type = get_type(parent->config,child);
        if(com->type == BH_COMPONENT_ERROR)
        {
	        fprintf(stderr, "bh_component_setup(): invalid component type: %s.\n", child);
	        result = BH_ERROR;
	        break;
        }

        if(!iniparser_find_entry(com->config,child))
        {
            fprintf(stderr,"Reference \"%s\" is not declared.\n",child);
	        result = BH_ERROR;
	        break;
        }

        snprintf(tmp, BH_COMPONENT_NAME_SIZE, "%s:impl", child);
        char *impl = iniparser_getstring(com->config, tmp, NULL);
        if(impl == NULL)
        {
            fprintf(stderr,"In section \"%s\" impl is not set.\n",child);
	        result = BH_ERROR;
	        break;
        }

        com->lib_handle = dlopen(impl, RTLD_NOW);
        if(com->lib_handle == NULL)
        {
            fprintf(stderr, "Error in [%s:impl]: %s\n", child, dlerror());
	        result = BH_ERROR;
	        break;
        }

        com->init = (bh_init)get_dlsym(com->lib_handle, child, com->type, "init");
        if(com->init == NULL)
        {
			fprintf(stderr, "Failed to load init function from child %s\n", child);
	        result = BH_ERROR;
	        break;
        }

        com->shutdown = (bh_shutdown)get_dlsym(com->lib_handle, child, com->type,
                                  "shutdown");
        if(com->shutdown == NULL)
        {
			fprintf(stderr, "Failed to load shutdown function from child %s\n", child);
	        result = BH_ERROR;
	        break;
        }

        com->execute = (bh_execute)get_dlsym(com->lib_handle, child, com->type,
                                 "execute");
        if(com->execute == NULL)
        {
			fprintf(stderr, "Failed to load execute function from child %s\n", child);
	        result = BH_ERROR;
	        break;
        }

        com->reg_func = (bh_reg_func)get_dlsym(com->lib_handle, child, com->type,
                                  "reg_func");
        if(com->reg_func == NULL)
        {
			fprintf(stderr, "Failed to load reg_func function from child %s\n", child);
	        result = BH_ERROR;
	        break;
        }

        child = strtok(NULL,",");
        ++(*count);
    }

	if (result != BH_SUCCESS)
	{
		for(c = 0; c < BH_COMPONENT_MAX_CHILDS; c++)
			if ((*children)[c] != NULL)
			{
				free((*children)[c]);
				(*children)[c] = NULL;
			}
		free(*children);
		*children = NULL;
	}
	else if(*count == 0)//No children.
    {
        free(*children);
        *children = NULL;
    }

    return result;
}

/* Retrieves an user-defined function.
 *
 * @self     The component.
 * @fun      Name of the function e.g. myfunc
 * @ret_func Pointer to the function (output)
 *           Is NULL if the function doesn't exist
 * @return Error codes (BH_SUCCESS)
 */
bh_error bh_component_get_func(bh_component *self, char *func,
                                     bh_userfunc_impl *ret_func)
{
    //First we search the libs in the config file to find the user-defined function.
    //Secondly we search the component's library.
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
                dlerror();//Clear old errors.
                *ret_func = (bh_userfunc_impl)dlsym(lib_handle, func);
                char *err = dlerror();
                if(err == NULL)
                    return BH_SUCCESS;
            }
            path = strtok(NULL,",");
        }
        free(lib_paths);
    }
    dlerror();//Clear old errors.
    *ret_func = (bh_userfunc_impl)dlsym(self->lib_handle, func);
    char *err = dlerror();
    if(err != NULL)
    {
        *ret_func = NULL;//Make sure it is NULL on error.
        fprintf(stderr, "Error when trying to load %s: %s\n", func, err);
        return BH_USERFUNC_NOT_SUPPORTED;
    }
    return BH_SUCCESS;
}

/* Frees the component.
 *
 * @return Error code (BH_SUCCESS).
 */
bh_error bh_component_free(bh_component *component)
{
    if(component->type == BH_BRIDGE)
        iniparser_freedict(component->config);
    else
        dlclose(component->lib_handle);
    free(component);
    return BH_SUCCESS;
}

/* Frees allocated data.
 *
 * @return Error code (BH_SUCCESS).
 */
bh_error bh_component_free_ptr(void* data)
{
    free(data);
    return BH_SUCCESS;
}

/* Look up a key in the config file
 *
 * @component The component.
 * @key       The key to lookup in the config file
 * @return    The value if found, otherwise NULL
 */
char* bh_component_config_lookup(bh_component *component, const char* key)
{
    char dictkey[BH_COMPONENT_NAME_SIZE];
    snprintf(dictkey, BH_COMPONENT_NAME_SIZE, "%s:%s", component->name, key);
    return iniparser_getstring(component->config, dictkey, NULL);
}
