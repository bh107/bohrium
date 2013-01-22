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

#ifdef __APPLE__
 //error.h is not required but the functions are known
 //the file is not found on OSX
#else
#include <error.h>
#endif
#include <bh.h>
#include <stdbool.h>

/* Text string for error code
 *
 * @error  Error code.
 * @return Text string.
 */
const char* bh_error_text(bh_error error)
{
    switch(error)
    {
    case BH_SUCCESS: 
        return "BH_SUCCESS";
    case BH_ERROR: 
        return "BH_ERROR";
    case BH_TYPE_NOT_SUPPORTED: 
        return "BH_TYPE_NOT_SUPPORTED";
    case BH_OUT_OF_MEMORY: 
        return "BH_OUT_OF_MEMORY";
    case BH_INST_NOT_SUPPORTED: 
        return "BH_INST_NOT_SUPPORTED";
    case BH_USERFUNC_NOT_SUPPORTED: 
        return "BH_USERFUNC_NOT_SUPPORTED";
    default:
        return "Error code unknown";
	}

}
