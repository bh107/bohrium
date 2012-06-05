/*
 * Copyright 2011 Mads R. B. Kristensen <madsbk@gmail.com>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#include <cphvb.h>
#include <iniparser.h>
#include <string.h>
#include <assert.h>

#ifdef _WIN32

#include <windows.h>
#include <dlfcn-win32.h>

#define HOME_INI_PATH "%APPDATA%\\cphvb\\config.ini"
#define SYSTEM_INI_PATH "%PROGRAMFILES%\\cphvb\\config.ini"

//We need a buffer for path expansion
char _expand_buffer1[MAX_PATH];
char _expand_buffer2[MAX_PATH];

//Nasty function renaming
#define snprintf _snprintf
#define strcasecmp _stricmp

#else

#include <dlfcn.h>
#include <limits.h>

#define HOME_INI_PATH "~/.cphvb/config.ini"
#define SYSTEM_INI_PATH "/opt/cphvb/config.ini"

//We need a buffer for path expansion
char _expand_buffer[PATH_MAX];

#endif


static cphvb_component_type get_type(dictionary *dict, const char *name)
{
    char tmp[CPHVB_COMPONENT_NAME_SIZE];
    snprintf(tmp, CPHVB_COMPONENT_NAME_SIZE, "%s:type", name);
    char *s = iniparser_getstring(dict, tmp, NULL);
    if(s == NULL)
    {
        fprintf(stderr,"In section \"%s\" type is not set. "\
                       "Should be bridge, vem or ve.\n",name);
        return CPHVB_COMPONENT_ERROR;
    }
    else
    {
        if(!strcasecmp(s, "bridge"))
            return CPHVB_BRIDGE;
        if(!strcasecmp(s, "vem"))
            return CPHVB_VEM;
        if(!strcasecmp(s, "ve"))
            return CPHVB_VE;
    }
    fprintf(stderr,"In section \"%s\" type is unknown: \"%s\" \n",
            name, s);
    return CPHVB_COMPONENT_ERROR;
}

static void *get_dlsym(void *handle, const char *name,
                       cphvb_component_type type, const char *fun)
{
    char tmp[1024];
    const char *stype;
    void *ret;
    if(type == CPHVB_BRIDGE)
        stype = "bridge";
    else if(type == CPHVB_VEM)
        stype = "vem";
    else if(type == CPHVB_VE)
        stype = "ve";
    else
    {
        fprintf(stderr, "get_dlsym - unknown component type.\n");
        return NULL;
    }

    snprintf(tmp, CPHVB_COMPONENT_NAME_SIZE, "cphvb_%s_%s_%s", stype, name, fun);
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
 * @return The root component in the configuration.
 */
cphvb_component *cphvb_component_setup(void)
{
    const char* homepath = HOME_INI_PATH;
    const char* syspath = SYSTEM_INI_PATH;

    cphvb_component *com = (cphvb_component*)malloc(sizeof(cphvb_component));
    const char *env;
    if(com == NULL)
    {
        fprintf(stderr, "cphvb_component_setup(): out of memory.\n");
        exit(CPHVB_OUT_OF_MEMORY);
    }

    //Clear memory so we do not have any random pointers
    memset(com, 0, sizeof(cphvb_component));

    strcpy(com->name, "bridge"); //The config root keyword.

    //The environment variable has precedence.
    env = getenv("CPHVB_CONFIG");
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
        fprintf(stderr, "Error: cphVB could not find the config file."
            " The search is:\n"
            "\t* The environment variable CPHVB_CONFIG.\n"
            "\t* The home directory \"%s\".\n"
            "\t* And system-wide \"%s\".\n", homepath, syspath);
        exit(-1);
    }

    com->config = iniparser_load(env);
    if(com->config == NULL)
        exit(-1);

    com->type = get_type(com->config, com->name);

    if(com->type != CPHVB_BRIDGE)
    {
        fprintf(stderr, "Error in the configuration: the root "
                        "component must be of type bridge.\n");
        exit(-1);
    }
    return com;
}

/* Retrieves the children components of the parent.
 *
 * @parent The parent component (input).
 * @count Number of children components(output).
 * @children Array of children components (output).
 * @return Error code (CPHVB_SUCCESS).
 */
cphvb_error cphvb_component_children(cphvb_component *parent, cphvb_intp *count,
                                     cphvb_component **children[])
{
    char tmp[CPHVB_COMPONENT_NAME_SIZE];
    char *child;
    *count = 0;
    snprintf(tmp, CPHVB_COMPONENT_NAME_SIZE, "%s:children",parent->name);
    char *tchildren = iniparser_getstring(parent->config, tmp, NULL);
    if(tchildren == NULL)
        exit(CPHVB_ERROR);

    *children = (cphvb_component**)malloc(CPHVB_COMPONENT_MAX_CHILDS * sizeof(cphvb_component *));
    if(*children == NULL)
    {
        fprintf(stderr, "cphvb_component_setup(): out of memory.\n");
        exit(CPHVB_OUT_OF_MEMORY);
    }

    //Handle one child at a time.
    child = strtok(tchildren,",");
    while(child != NULL && *count < CPHVB_COMPONENT_MAX_CHILDS)
    {
        (*children)[*count] = (cphvb_component*)malloc(sizeof(cphvb_component));
        cphvb_component *com = (*children)[*count];

        //Save component name.
        strncpy(com->name, child, CPHVB_COMPONENT_NAME_SIZE);
        //Save configuration dictionary.
        com->config = parent->config;
        //Save component type.
        com->type = get_type(parent->config,child);
        if(com->type == CPHVB_COMPONENT_ERROR)
        {
            exit(CPHVB_ERROR);
        }

        if(!iniparser_find_entry(com->config,child))
        {
            fprintf(stderr,"Reference \"%s\" is not declared.\n",child);
            exit(CPHVB_ERROR);
        }

        snprintf(tmp, CPHVB_COMPONENT_NAME_SIZE, "%s:impl", child);
        char *impl = iniparser_getstring(com->config, tmp, NULL);
        if(impl == NULL)
        {
            fprintf(stderr,"In section \"%s\" impl is not set.\n",child);
            exit(CPHVB_ERROR);
        }

        com->lib_handle = dlopen(impl, RTLD_NOW);
        if(com->lib_handle == NULL)
        {
            fprintf(stderr, "Error in [%s:impl]: %s\n", child, dlerror());
            exit(CPHVB_ERROR);
        }

        com->init = (cphvb_init)get_dlsym(com->lib_handle, child, com->type, "init");
        if(com->init == NULL)
            exit(CPHVB_ERROR);

        com->shutdown = (cphvb_shutdown)get_dlsym(com->lib_handle, child, com->type,
                                  "shutdown");
        if(com->shutdown == NULL)
            exit(CPHVB_ERROR);

        com->execute = (cphvb_execute)get_dlsym(com->lib_handle, child, com->type,
                                 "execute");
        if(com->execute == NULL)
            exit(CPHVB_ERROR);

        com->reg_func = (cphvb_reg_func)get_dlsym(com->lib_handle, child, com->type,
                                  "reg_func");
        if(com->reg_func == NULL)
            exit(CPHVB_ERROR);

        if(com->type == CPHVB_VEM)//VEM functions only.
        {
            com->create_array = (cphvb_create_array)get_dlsym(com->lib_handle, child,
                                          com->type, "create_array");
            if(com->create_array == NULL)
                exit(CPHVB_ERROR);
        }
        child = strtok(NULL,",");
        ++(*count);
    }

    if(*count == 0)//No children.
    {
        free(*children);
        *children = NULL;
    }
    
    return CPHVB_SUCCESS;
}

/* Retrieves an user-defined function.
 *
 * @self     The component.
 * @fun      Name of the function e.g. myfunc
 * @ret_func Pointer to the function (output)
 *           Is NULL if the function doesn't exist
 * @return Error codes (CPHVB_SUCCESS)
 */
cphvb_error cphvb_component_get_func(cphvb_component *self, char *func,
                                     cphvb_userfunc_impl *ret_func)
{
    //First we search the libs in the config file to find the user-defined function.
    //Secondly we search the component's library. 
    char *lib_paths = cphvb_component_config_lookup(self,"libs");
    if(lib_paths != NULL)
    {
        //Handle one library path at a time.
        char *path = strtok(lib_paths,",");
        while(path != NULL)
        {
            void *lib_handle = dlopen(path, RTLD_NOW);
            if(lib_handle != NULL)
            {
                dlerror();//Clear old errors.
                *ret_func = (cphvb_userfunc_impl)dlsym(lib_handle, func);
                char *err = dlerror();
                if(err == NULL)
                    return CPHVB_SUCCESS;
            }
            path = strtok(NULL,",");
        }
    }
    dlerror();//Clear old errors.
    *ret_func = (cphvb_userfunc_impl)dlsym(self->lib_handle, func);
    char *err = dlerror();
    if(err != NULL)
    {
        *ret_func = NULL;//Make sure it is NULL on error.
        return CPHVB_USERFUNC_NOT_SUPPORTED;
    }
    return CPHVB_SUCCESS;
}

/* Frees the component.
 *
 * @return Error code (CPHVB_SUCCESS).
 */
cphvb_error cphvb_component_free(cphvb_component *component)
{
    if(component->type == CPHVB_BRIDGE)
        iniparser_freedict(component->config);
    else
        dlclose(component->lib_handle);
    free(component);
    return CPHVB_SUCCESS;
}

/* Frees allocated data.
 *
 * @return Error code (CPHVB_SUCCESS).
 */
cphvb_error cphvb_component_free_ptr(void* data)
{
    free(data);
    return CPHVB_SUCCESS;
}

/* Trace an array creation.
 *
 * @self The component.
 * @ary  The array to trace.
 * @return Error code (CPHVB_SUCCESS).
 */
cphvb_error cphvb_component_trace_array(cphvb_component *self, cphvb_array *ary)
{
    int i;
    FILE *f = fopen("/tmp/cphvb_trace.ary", "a");
    fprintf(f,"array: %p;\t ndim: %ld;\t shape:", ary, ary->ndim);
    for(i=0; i<ary->ndim; ++i)
        fprintf(f," %ld", ary->shape[i]);
    fprintf(f,";\t stride:");
    for(i=0; i<ary->ndim; ++i)
        fprintf(f," %ld", ary->stride[i]);
    fprintf(f,";\t start: %ld;\t base: %p;\n",ary->start, ary->base);

    fclose(f);
    return CPHVB_SUCCESS;
}

/* Trace an instruction.
 *
 * @self The component.
 * @inst  The instruction to trace.
 * @return Error code (CPHVB_SUCCESS).
 */
cphvb_error cphvb_component_trace_inst(cphvb_component *self, cphvb_instruction *inst)
{
    int i;
    cphvb_intp nop;
    cphvb_array *ops[CPHVB_MAX_NO_OPERANDS];

    FILE *f = fopen("/tmp/cphvb_trace.inst", "a");

    fprintf(f,"%s\t", cphvb_opcode_text(inst->opcode));

    if(inst->opcode == CPHVB_USERFUNC)
    {
        nop = inst->userfunc->nout + inst->userfunc->nin;
        for(i=0; i<nop; ++i)
            ops[i] = inst->userfunc->operand[i];
    }
    else
    {
        nop = cphvb_operands(inst->opcode);
        for(i=0; i<nop; ++i)
            ops[i] = inst->operand[i];
    }
    for(i=0; i<nop; ++i)
        fprintf(f," \t%p", ops[i]);

    fprintf(f,"\n");

    fclose(f);
    return CPHVB_SUCCESS;
}

/* Look up a key in the config file 
 *
 * @component The component.
 * @key       The key to lookup in the config file
 * @return    The value if found, otherwise NULL
 */
char* cphvb_component_config_lookup(cphvb_component *component, const char* key)
{
    char dictkey[CPHVB_COMPONENT_NAME_SIZE];
    snprintf(dictkey, CPHVB_COMPONENT_NAME_SIZE, "%s:%s", component->name, key);
    return iniparser_getstring(component->config, dictkey, NULL);    
}
