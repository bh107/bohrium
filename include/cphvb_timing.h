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

#ifndef __CPHVB_TIMING_H
#define __CPHVB_TIMING_H

#include "cphvb_type.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CPHVB_TIMING_MAX 32

typedef unsigned long long cphvb_time;
typedef struct
{
    char *names[CPHVB_TIMING_MAX];
    cphvb_time times[CPHVB_TIMING_MAX];
    cphvb_intp count;
}cphvb_timing;


/* Start the timer.
 *
 * @timer The timer.
 * @name Name of the timing.
 * @return The id that should be passed to cphvb_timing_stop().
 */
cphvb_intp cphvb_timing_init(cphvb_timing *timer, char *name);

/* Start the timer.
 *
 * @return The current time.
 */
cphvb_time cphvb_timing_start(void);

/* Stop the timer and save the result.
 *
 * @timer The timer.
 * @id The id that was returned by cphvb_timing_init().
 * @time The timed returned by cphvb_timing_start().
 */
void cphvb_timing_stop(cphvb_timing *timer, cphvb_intp id,
                       cphvb_time time);




#ifdef __cplusplus
}
#endif

#endif
