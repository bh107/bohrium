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
#ifndef __BH_FILTER_PPRINT_H
#define __BH_FILTER_PPRINT_H

#include <bh.h>
#include "pprint_filter.h"

#ifdef __cplusplus
extern "C" {
#endif

DLLEXPORT bh_error bh_filter_pprint_init(bh_component *self);
DLLEXPORT bh_error bh_filter_pprint_execute(bh_ir* bhir);
DLLEXPORT bh_error bh_filter_pprint_shutdown(void);
DLLEXPORT bh_error bh_filter_pprint_reg_func(const char *fun, bh_intp *id);

#ifdef __cplusplus
}
#endif

#endif
