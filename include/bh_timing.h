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

#ifndef __CPHVB_TIMING_H
#define __CPHVB_TIMING_H

#include "bh_type.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CPHVB_TIMING_MAX 32

typedef unsigned long long bh_time;
typedef struct
{
    char *names[CPHVB_TIMING_MAX];
    bh_time times[CPHVB_TIMING_MAX];
    bh_intp count;
}bh_timing;


/* Start the timer.
 *
 * @timer The timer.
 * @name Name of the timing.
 * @return The id that should be passed to bh_timing_stop().
 */
bh_intp bh_timing_init(bh_timing *timer, char *name);

/* Start the timer.
 *
 * @return The current time.
 */
bh_time bh_timing_start(void);

/* Stop the timer and save the result.
 *
 * @timer The timer.
 * @id The id that was returned by bh_timing_init().
 * @time The timed returned by bh_timing_start().
 */
void bh_timing_stop(bh_timing *timer, bh_intp id,
                       bh_time time);




#ifdef __cplusplus
}
#endif

#endif
