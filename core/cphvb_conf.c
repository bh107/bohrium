/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <cphvb_conf.h>
#include <iniparser.h>
#include <string.h>
#include <dlfcn.h>


static dictionary *load_conf(void)
{
    char *env;
    env = getenv("CPHVB_CONFIG");
    if(env == NULL)
        env = "config.ini";

    return iniparser_load(env);

}

static cphvb_component get_type(dictionary *dict, const char *name)
{
    char tmp[1024];
    sprintf(tmp, "%s:type", name);
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
                       cphvb_component type, const char *fun)
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

    sprintf(tmp, "cphvb_%s_%s_%s", stype, name, fun);
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


cphvb_error cphvb_conf_children(const char *component_name, cphvb_interface *if_vem)
{
    dictionary *dict = load_conf();
    char tmp[1024];
    char *child;
    sprintf(tmp, "%s:children",name);
    char *children = iniparser_getstring(dict, tmp, NULL);
    if(children == NULL)
        return CPHVB_ERROR;

    //Handle one child at a time.
    child = strtok(children,",");
    while(child != NULL)
    {
        cphvb_component type = get_type(dict,child);
        if(type == CPHVB_COMPONENT_ERROR)
            return CPHVB_ERROR;

        if(!iniparser_find_entry(dict,child))
        {
            fprintf(stderr,"Reference \"%s\" is not declared.\n",child);
            return CPHVB_ERROR;
        }

        sprintf(tmp, "%s:impl", child);
        char *impl = iniparser_getstring(dict, tmp, NULL);
        if(impl == NULL)
        {
            fprintf(stderr,"In section \"%s\" impl is not set.\n",child);
            return CPHVB_ERROR;
        }

        void *handle = dlopen(impl, RTLD_NOW);
        if(handle == NULL)
        {
            fprintf(stderr, "Error in [%s:impl]: %s\n", child, dlerror());
            return CPHVB_ERROR;
        }

        if_vem->init = get_dlsym(handle, child, type, "init");
        if(if_vem->init == NULL)
            return CPHVB_ERROR;
        if_vem->shutdown = get_dlsym(handle, child, type, "shutdown");
        if(if_vem->shutdown == NULL)
            return CPHVB_ERROR;
        if_vem->execute = get_dlsym(handle, child, type, "execute");
        if(if_vem->execute == NULL)
            return CPHVB_ERROR;

        if(type == CPHVB_VEM)//VEM functions only.
        {
            if_vem->create_array = get_dlsym(handle, child, type, "create_array");
            if(if_vem->create_array == NULL)
                return CPHVB_ERROR;
        }

        child = strtok(NULL,",");
    }


    return CPHVB_SUCCESS;
}


