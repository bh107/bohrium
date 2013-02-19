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

#ifndef __BOHRIUM_BRIDGE_CPP_FUNCTIONS
#define __BOHRIUM_BRIDGE_CPP_FUNCTIONS
#include "bh.h"

namespace bh {


    template <typename T>
    Vector<T> & pow ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_POWER, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & abs ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_ABSOLUTE, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & max ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_MAXIMUM, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & min ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_MINIMUM, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & sin ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_SIN, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & cos ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_COS, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & tan ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_TAN, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & sinh ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_SINH, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & cosh ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_COSH, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & tanh ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_TANH, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & exp ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_EXP, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & exp2 ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_EXP2, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & expm1 ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_EXPM1, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & log ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_LOG, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & log2 ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_LOG2, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & log10 ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_LOG10, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & log1p ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_LOG1P, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & sqrt ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_SQRT, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & ceil ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_CEIL, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & trunc ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_TRUNC, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & floor ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_FLOOR, vector, vector );
        return vector;
    }


    template <typename T>
    Vector<T> & rint ( Vector<T> & vector )
    {
        std::cout << vector << std::endl;
        enqueue_aa( (bh_opcode) BH_RINT, vector, vector );
        return vector;
    }


}

#endif

