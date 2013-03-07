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

#include "iterator.hpp"

namespace bh {

template <typename T>
class Vector {
public:
    // Types:
    typedef Vector_iter<T> iterator;

    Vector( Vector const& vector );
    Vector( int d0 );
    Vector( int d0, int d1 );

    int getKey() const;
    bool isTemp() const;
    void setTemp(bool is_temp);

    ~Vector();

    //
    // Operators: 
    //
    // =, [], (), -> must be "internal" (nonstatic member functions) and thus declared here.
    //
    // Implementations / definitions of operator-overloads are provided in:
    // 
    // - vector.hpp:    defined / implemented manually.
    // - operators.hpp: defined / implemented by code-generator.
    //
    Vector& operator=( const T rhs );
    Vector& operator=( Vector & rhs );
    Vector& operator++();
    Vector& operator++( int );
    Vector& operator--();
    Vector& operator--( int );

    Vector& operator+=( const T rhs );
    Vector& operator+=( Vector & rhs );

    iterator begin();
    iterator end();

private:

    int key;
    bool is_temp;
    bool synced;

};

}

#include "vector.hpp"       // Vector (De)Constructor.
#include "state.hpp"        // Communication with Bohrium runtime
#include "operators.hpp"    // Vector operations via operator-overloads.
#include "functions.hpp"    // Vector operations via functions.
#include "sugar.hpp"        // Pretty print functions and the like...

#endif
