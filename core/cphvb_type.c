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

#include <cphvb_type.h>
#include <cphvb.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Byte size for type
 *
 * @type   Type code
 * @return Byte size
 */
int cphvb_type_size(cphvb_type type)
{
    switch(type)
    {
    case CPHVB_BOOL:
        return 1;
    case CPHVB_INT8:
        return 1;
    case CPHVB_INT16:
        return 2;
    case CPHVB_INT32:
        return 4;
    case CPHVB_INT64:
        return 8;
    case CPHVB_UINT8:
        return 1;
    case CPHVB_UINT16:
        return 2;
    case CPHVB_UINT32:
        return 4;
    case CPHVB_UINT64:
        return 8;
    case CPHVB_FLOAT16:
        return 2;
    case CPHVB_FLOAT32:
        return 4;
    case CPHVB_FLOAT64:
        return 8;
    case CPHVB_COMPLEX64:
        return 8;
    case CPHVB_COMPLEX128:
        return 16;
    case CPHVB_UNKNOWN:
        return -1;
    default:
        return -1;
	}
}

/* Text string for type
 *
 * @type   Type code.
 * @return Text string.
 */
const char* cphvb_type_text(cphvb_type type)
{
    switch(type)
    {
    case CPHVB_BOOL:
        return "CPHVB_BOOL";
    case CPHVB_INT8:
        return "CPHVB_INT8";
    case CPHVB_INT16:
        return "CPHVB_INT16";
    case CPHVB_INT32:
        return "CPHVB_INT32";
    case CPHVB_INT64:
        return "CPHVB_INT64";
    case CPHVB_UINT8:
        return "CPHVB_UINT8";
    case CPHVB_UINT16:
        return "CPHVB_UINT16";
    case CPHVB_UINT32:
        return "CPHVB_UINT32";
    case CPHVB_UINT64:
        return "CPHVB_UINT64";
    case CPHVB_FLOAT16:
        return "CPHVB_FLOAT16";
    case CPHVB_FLOAT32:
        return "CPHVB_FLOAT32";
    case CPHVB_FLOAT64:
        return "CPHVB_FLOAT64";
    case CPHVB_COMPLEX64:
        return "CPHVB_COMPLEX64";
    case CPHVB_COMPLEX128:
        return "CPHVB_COMPLEX128";
    case CPHVB_UNKNOWN:
        return "CPHVB_UNKNOWN";
    default:
        return "Unknown type";
    }
}

#ifdef __cplusplus
}
#endif
