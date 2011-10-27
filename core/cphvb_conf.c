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

static dictionary *load_conf(void)
{
    char *env;
    env = getenv("CPHVB_CONFIG");
    if(env == NULL)
        env = "config.ini";

    return iniparser_load(env);

}

cphvb_error cphvb_conf_children(cphvb_vem_interface *if_vem)
{
    dictionary *dict = load_conf();
    iniparser_dump(dict,stdout);
    return CPHVB_SUCCESS;
}


