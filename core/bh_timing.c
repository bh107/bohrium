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

#include <sys/time.h>
#include <bh_timing.h>

/* Start the timer.
 *
 * @timer The timer.
 * @name Name of the timing.
 * @return The id that should be passed to bh_timing_stop().
 */
bh_intp bh_timing_init(bh_timing *timer, char *name)
{
    bh_intp id = ++timer->count;
    timer->names[id] = name;
    timer->times[id] = 0;
    return id;
}

/* Start the timer.
 *
 * @return The current time.
 */
bh_time bh_timing_start(void)
{
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    return (unsigned long long) tv.tv_usec +
           (unsigned long long) tv.tv_sec * 1000000;
}

/* Stop the timer and save the result.
 *
 * @timer The timer.
 * @id The id that was returned by bh_timing_init().
 * @time The timed returned by bh_timing_start().
 */
void bh_timing_stop(bh_timing *timer, bh_intp id,
                       bh_time time)
{
    struct timeval tv;
    struct timezone tz;
    gettimeofday(&tv, &tz);
    bh_time delta = ((unsigned long long) tv.tv_usec +
                        (unsigned long long) tv.tv_sec * 1000000) - time;
    //Save the timing.
    timer->times[id] += delta;
}
