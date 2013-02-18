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

#ifndef __BH_TIMING_H
#define __BH_TIMING_H

#include "bh_type.h"

#ifdef __cplusplus
extern "C" {
#endif


//When BH_TIMING_SUM is defined we only record timing sums.
#ifdef BH_TIMING_SUM
    #define BH_TIMING
#endif

//Only when BH_TIMING is defined will the bh_timing* functions do anything.
#ifdef BH_TIMING
    #define bh_timing_new(name) (_bh_timing_new(name))
    #define bh_timing() (_bh_timing()) 
    #define bh_timing_dump_all() (_bh_timing_dump_all())
    #ifdef BH_TIMING_SUM
        #define bh_timing_save(id,start,end) (_bh_timing_save_sum(id,start,end))
    #else
        #define bh_timing_save(id,start,end) (_bh_timing_save(id,start,end))
    #endif
#else
    #define bh_timing_new(name) ((bh_intp)0);
    #define bh_timing_save(id,start,end) do{(void)(id);(void)(start);(void)(end);} while (0)
    #define bh_timing() ((bh_uint64)0)
    #define bh_timing_dump_all()  {}
#endif

/* Initiate new timer object.
 *
 * @name Name of the timing.
 * @return The timer ID.
 */
bh_intp _bh_timing_new(const char *name);


/* Save a timing.
 *
 * @id     The ID of the timing.
 * @start  The start time in micro sec.
 * @end    The end time in micro sec.
 */
void _bh_timing_save(bh_intp id, bh_uint64 start, bh_uint64 end);


/* Save the sum of a timing. 
 *
 * @id     The ID of the timing.
 * @start  The start time in micro sec.
 * @end    The end time in micro sec.
 */
void _bh_timing_save_sum(bh_intp id, bh_uint64 start, bh_uint64 end);


/* Get time.
 *
 * @return The current time.
 */
bh_uint64 _bh_timing(void);


/* Dumps all timings to a file in the working directory.
 *
 */
void _bh_timing_dump_all(void);


#ifdef __cplusplus
}
#endif

#endif
