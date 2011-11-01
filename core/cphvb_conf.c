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

cphvb_error cphvb_conf_children(char *name, cphvb_vem_interface *if_vem)
{
    dictionary *dict = load_conf();
    iniparser_dump(dict,stdout);
    char tmp[1024];
    char *child;
    sprintf(tmp, "%s:children",name);
    char *children = iniparser_getstring(dict, tmp, NULL);

    printf("My Children: %s\n",children);

    //Handle one child at a time.
    child = strtok(children,",");
    while(child != NULL)
    {
        printf("Child: %s\n",child);

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

        sprintf(tmp, "cphvb_vem_%s_init", child);
        dlerror();//Clear old errors.
        if_vem->init = dlsym(handle, tmp);
        char *err = dlerror();
        if(err != NULL)
        {
            fprintf(stderr, "%s\n", err);
            return CPHVB_ERROR;
        }
        sprintf(tmp, "cphvb_vem_%s_shutdown", child);
        dlerror();//Clear old errors.
        if_vem->shutdown = dlsym(handle, tmp);
        err = dlerror();
        if(err != NULL)
        {
            fprintf(stderr, "%s\n", err);
            return CPHVB_ERROR;
        }
        sprintf(tmp, "cphvb_vem_%s_execute", child);
        dlerror();//Clear old errors.
        if_vem->execute = dlsym(handle, tmp);
        err = dlerror();
        if(err != NULL)
        {
            fprintf(stderr, "%s\n", err);
            return CPHVB_ERROR;
        }
        sprintf(tmp, "cphvb_vem_%s_create_array", child);
        dlerror();//Clear old errors.
        if_vem->create_array = dlsym(handle, tmp);
        err = dlerror();
        if(err != NULL)
        {
            fprintf(stderr, "%s\n", err);
            return CPHVB_ERROR;
        }
        sprintf(tmp, "cphvb_vem_%s_instruction_check", child);
        dlerror();//Clear old errors.
        if_vem->instruction_check = dlsym(handle, tmp);
        err = dlerror();
        if(err != NULL)
        {
            fprintf(stderr, "%s\n", err);
            return CPHVB_ERROR;
        }

        child = strtok(NULL,",");
    }


    return CPHVB_SUCCESS;
}


