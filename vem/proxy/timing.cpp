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

#include <cassert>
#include <cstring>
#include <iostream>
#include <bh.h>
#include "timing.h"
#include <set>

static int sleep = -1;

//Sleep a period based on BH_VEM_PROXY_SLEEP (in ms)
void timing_sleep(void)
{
    if(sleep == -1)
    {
        const char *str = getenv("BH_VEM_PROXY_SLEEP");
        if(str == NULL)
            sleep = 0;
        else
            sleep = atoi(str);
        printf("sleep enabled: %dms\n", sleep);
    }
    if(sleep == 0)
        return;

    struct timespec tim, tim2;
    tim.tv_sec = sleep/1000000;
    tim.tv_nsec = sleep%1000000 * 1000;

    if(nanosleep(&tim , &tim2) < 0 )
        printf("Nano sleep system call failed \n");
}

