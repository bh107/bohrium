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
#ifndef __BH_ERROR_H
#define __BH_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
enum /* bh_error */
{
    BH_SUCCESS,               // General success
    BH_ERROR,                 // Fatal error 
    BH_TYPE_NOT_SUPPORTED,    // Data type not supported
    BH_OUT_OF_MEMORY,         // Out of memory
    BH_INST_NOT_SUPPORTED,    // Instruction not supported
    BH_USERFUNC_NOT_SUPPORTED // User-defined function not supported
};


#ifdef __cplusplus
}
#endif

#endif
