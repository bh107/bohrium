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

#ifndef __BH_SIGNAL_H
#define __BH_SIGNAL_H

#ifndef _XOPEN_SOURCE
    #define _XOPEN_SOURCE
#endif

#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <signal.h>
#include <ucontext.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif


int init_signal(void);

int attach_signal(signed long idx,//callback id
                  uintptr_t start,//Start memory address
                  long int size,  //Size of the memory (in bytes)
                  void (*callback)(unsigned long, uintptr_t));//The callback function

int detach_signal(signed long idx, void (*callback));

#ifdef __cplusplus
}
#endif

#endif
