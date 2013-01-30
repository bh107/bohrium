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

#ifndef BH_TIMING
    #define bh_timing_new(name) ((bh_intp)0);
    #define bh_timing_save(id,start,end) do{(void)(id);(void)(start);(void)(end);} while (0)
    #define bh_timing() ((bh_uint64)0)
    #define bh_timing_dump_all()  {}
#else
    /* Initiate new timer object.
     *
     * @name Name of the timing.
     * @return The timer ID.
     */
    #define bh_timing_new(name) (_bh_timing_new(name))
    bh_intp _bh_timing_new(const char *name);


    /* Save a timing.
     *
     * @id     The ID of the timing.
     * @start  The start time in micro sec.
     * @end    The end time in micro sec.
     */
    #define bh_timing_save(id,start,end) (_bh_timing_save(id,start,end))
    void _bh_timing_save(bh_intp id, bh_uint64 start, bh_uint64 end);


    /* Get time.
     *
     * @return The current time.
     */
    #define bh_timing() (_bh_timing()) 
    bh_uint64 _bh_timing(void);


    /* Dumps all timings to a file in the working directory.
     *
     */
    #define bh_timing_dump_all() (_bh_timing_dump_all())
    void _bh_timing_dump_all(void);
#endif


#ifdef __cplusplus
}
#endif

#endif
