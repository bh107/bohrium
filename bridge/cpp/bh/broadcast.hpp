/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://bohrium.bitbucket.org

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
#ifndef __BOHRIUM_BRIDGE_CPP
#define __BOHRIUM_BRIDGE_CPP
#include "bh.h"

namespace bh {

template <typename T>
inline
Vector<T> & broadcast( Vector<T> & from, Vector<T> & to )
{

}

/**
 *  Determine compatibility of two operands.
 *
 *  @return 0 = Not in any way.
 *          1 = Ready to rock
 *          2 = op1 is broadcastable to op2
 *          3 = op2 is broadcastable to op1
 */
template <typename T>
inline
int compatible( Vector<T> & op1, Vector<T> op2 )
{
    int c = 0;
}

}
#endif
