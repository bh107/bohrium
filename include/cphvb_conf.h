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

#ifndef __CPHVB_CONF_H
#define __CPHVB_CONF_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cphvb.h>
#include <cphvb_vem.h>
#include <iniparser.h>

typedef struct
{
    cphvb_vem_init init;
    cphvb_vem_shutdown shutdown;
    cphvb_vem_execute execute;
    cphvb_vem_create_array create_array;
    cphvb_vem_instruction_check instruction_check;
} cphvb_vem_interface;


cphvb_error cphvb_conf_children(char *name, cphvb_vem_interface *if_vem);



#ifdef __cplusplus
}
#endif

#endif
