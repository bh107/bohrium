/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
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

//
//  WARN:   This file is generated; changes to it will be overwritten.
//          If you wish to change its functionality then change the code-generator for this file.
//          Take a look at: codegen/README
//

#ifndef __BOHRIUM_BRIDGE_CPP_FUNCTIONS
#define __BOHRIUM_BRIDGE_CPP_FUNCTIONS
#include "bh.h"

namespace bh {

template <typename T>
Vector<T> & pow ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_POWER, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & abs ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_ABSOLUTE, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & max ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_MAXIMUM, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & min ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_MINIMUM, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & sin ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_SIN, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & cos ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_COS, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & tan ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_TAN, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & sinh ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_SINH, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & cosh ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_COSH, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & tanh ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_TANH, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & exp ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_EXP, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & exp2 ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_EXP2, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & expm1 ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_EXPM1, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & log ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_LOG, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & log2 ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_LOG2, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & log10 ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_LOG10, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & log1p ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_LOG1P, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & sqrt ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_SQRT, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & ceil ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_CEIL, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & trunc ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_TRUNC, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & floor ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_FLOOR, *vector, rhs );
    return *vector;
}

template <typename T>
Vector<T> & rint ( Vector<T> & rhs )
{
    Vector<T>* vector = new Vector<T>( rhs );

    Runtime::instance()->enqueue( (bh_opcode) BH_RINT, *vector, rhs );
    return *vector;
}

}
#endif

