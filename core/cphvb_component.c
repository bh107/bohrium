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

#include <cphvb_component.h>
#include <iniparser.h>
#include <string.h>
#include <dlfcn.h>
#include <assert.h>

static cphvb_com_type get_type(dictionary *dict, const char *name)
{
    char tmp[CPHVB_COM_NAME_SIZE];
    snprintf(tmp, CPHVB_COM_NAME_SIZE, "%s:type", name);
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
                       cphvb_com_type type, const char *fun)
{
    char tmp[1024];
    char *stype;
    void *ret;
    if(type == CPHVB_BRIDGE)
        stype = "bridge";
    else if(type == CPHVB_VEM)
        stype = "vem";
    else if(type == CPHVB_VE)
        stype = "ve";
    else
    {
        fprintf(stderr, "get_dlsym - unknown componet type.\n");
        return NULL;
    }

    snprintf(tmp, CPHVB_COM_NAME_SIZE, "cphvb_%s_%s_%s", stype, name, fun);
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
cphvb_com *cphvb_com_setup(void)
{
    cphvb_com *com = malloc(sizeof(cphvb_com));
    char *env;
    if(com == NULL)
    {
        fprintf(stderr, "cphvb_com_setup(): out of memory.\n");
        exit(CPHVB_OUT_OF_MEMORY);
    }
    strcpy(com->name, "bridge"); //The config root keyword.

    env = getenv("CPHVB_CONFIG");
    if(env == NULL)
        env = "config.ini";

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
cphvb_error cphvb_com_children(cphvb_com *parent, cphvb_intp *count,
                               cphvb_com **children[])
{
    char tmp[CPHVB_COM_NAME_SIZE];
    char *child;
    *count = 0;
    snprintf(tmp, CPHVB_COM_NAME_SIZE, "%s:children",parent->name);
    char *tchildren = iniparser_getstring(parent->config, tmp, NULL);
    if(tchildren == NULL)
        exit(CPHVB_ERROR);

    *children = malloc(10 * sizeof(cphvb_com *));
    if(*children == NULL)
    {
        fprintf(stderr, "cphvb_com_setup(): out of memory.\n");
        exit(CPHVB_OUT_OF_MEMORY);
    }

    //Handle one child at a time.
    child = strtok(tchildren,",");
    while(child != NULL)
    {
        (*children)[*count] = malloc(sizeof(cphvb_com));
        cphvb_com *com = (*children)[*count];

        //Save component name.
        strncpy(com->name, child, CPHVB_COM_NAME_SIZE);
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

        snprintf(tmp, CPHVB_COM_NAME_SIZE, "%s:impl", child);
        char *impl = iniparser_getstring(com->config, tmp, NULL);
        if(impl == NULL)
        {
            fprintf(stderr,"In section \"%s\" impl is not set.\n",child);
            exit(CPHVB_ERROR);
        }

        void *handle = dlopen(impl, RTLD_NOW);
        if(handle == NULL)
        {
            fprintf(stderr, "Error in [%s:impl]: %s\n", child, dlerror());
            exit(CPHVB_ERROR);
        }

        com->init = get_dlsym(handle, child, com->type, "init");
        if(com->init == NULL)
            exit(CPHVB_ERROR);
        com->shutdown = get_dlsym(handle, child, com->type, "shutdown");
        if(com->shutdown == NULL)
            exit(CPHVB_ERROR);
        com->execute = get_dlsym(handle, child, com->type, "execute");
        if(com->execute == NULL)
            exit(CPHVB_ERROR);

        if(com->type == CPHVB_VEM)//VEM functions only.
        {
            com->create_array = get_dlsym(handle, child, com->type,
                                          "create_array");
            if(com->create_array == NULL)
                exit(CPHVB_ERROR);
            com->instruction_check = get_dlsym(handle, child, com->type,
                                               "instruction_check");
            if(com->instruction_check == NULL)
                exit(CPHVB_ERROR);
        }
        child = strtok(NULL,",");
        ++(*count);
    }

    if(*count == 0)//No children.
        free(*children);

    return CPHVB_SUCCESS;
}


/* Frees the component.
 *
 * @return Error code (CPHVB_SUCCESS).
 */
cphvb_error cphvb_com_free(cphvb_com *component)
{
    if(component->type == CPHVB_BRIDGE)
        iniparser_freedict(component->config);
    free(component);
    return CPHVB_SUCCESS;
}



