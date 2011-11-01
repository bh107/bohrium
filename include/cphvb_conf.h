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
#include <cphvb_interface.h>
#include <iniparser.h>
#include <cphvb_error.h>


cphvb_error cphvb_conf_children(const char *name, cphvb_interface *if_vem);



#ifdef __cplusplus
}
#endif

#endif
