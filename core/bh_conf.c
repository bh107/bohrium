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

#include <bh_conf.h>
#include <iniparser.h>
#include <string.h>
#include <dlfcn.h>


static dictionary *load_conf(void)
{
    char *env;
    env = getenv("BH_CONFIG");
    if(env == NULL)
        env = "config.ini";

    return iniparser_load(env);

}

static bh_component get_type(dictionary *dict, const char *name)
{
    char tmp[1024];
    sprintf(tmp, "%s:type", name);
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
        if(!strcasecmp(s, "filter"))
            return BH_FILTER;
    }
    fprintf(stderr,"In section \"%s\" type is unknown: \"%s\" \n",
            name, s);
    return BH_COMPONENT_ERROR;
}

static void *get_dlsym(void *handle, const char *name,
                       bh_component type, const char *fun)
{
    char tmp[1024];
    char *stype;
    void *ret;
    if(type == BH_BRIDGE)
        stype = "bridge";
    else if(type == BH_VEM)
        stype = "vem";
    else if(type == BH_VE)
        stype = "ve";
    else if(type == BH_FILTER)
        stype = "filter";
    else
    {
        fprintf(stderr, "get_dlsym - unknown componet type.\n");
        return NULL;
    }

    sprintf(tmp, "bh_%s_%s_%s", stype, name, fun);
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


bh_error bh_conf_children(const char *component_name, bh_interface *if_vem)
{
    dictionary *dict = load_conf();
    char tmp[1024];
    char *child;
    sprintf(tmp, "%s:children",name);
    char *children = iniparser_getstring(dict, tmp, NULL);
    if(children == NULL)
        return BH_ERROR;

    //Handle one child at a time.
    child = strtok(children,",");
    while(child != NULL)
    {
        bh_component type = get_type(dict,child);
        if(type == BH_COMPONENT_ERROR)
            return BH_ERROR;

        if(!iniparser_find_entry(dict,child))
        {
            fprintf(stderr,"Reference \"%s\" is not declared.\n",child);
            return BH_ERROR;
        }

        sprintf(tmp, "%s:impl", child);
        char *impl = iniparser_getstring(dict, tmp, NULL);
        if(impl == NULL)
        {
            fprintf(stderr,"In section \"%s\" impl is not set.\n",child);
            return BH_ERROR;
        }

        void *handle = dlopen(impl, RTLD_NOW);
        if(handle == NULL)
        {
            fprintf(stderr, "Error in [%s:impl]: %s\n", child, dlerror());
            return BH_ERROR;
        }

        if_vem->init = get_dlsym(handle, child, type, "init");
        if(if_vem->init == NULL)
            return BH_ERROR;
        if_vem->shutdown = get_dlsym(handle, child, type, "shutdown");
        if(if_vem->shutdown == NULL)
            return BH_ERROR;
        if_vem->execute = get_dlsym(handle, child, type, "execute");
        if(if_vem->execute == NULL)
            return BH_ERROR;

        if(type == BH_VEM)//VEM functions only.
        {
            if_vem->create_array = get_dlsym(handle, child, type, "create_array");
            if(if_vem->create_array == NULL)
                return BH_ERROR;
        }

        child = strtok(NULL,",");
    }


    return BH_SUCCESS;
}


