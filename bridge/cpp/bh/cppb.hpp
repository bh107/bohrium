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

#define BH_CPP_QUEUE_MAX 1024
#include "iterator.hpp"

namespace bh {

template <typename T>
class Operand {
public:
    Operand();
    ~Operand();

    // Types:
    typedef Operand_iter<T> iterator;

    int getKey() const;
    bool isTemp() const;
    void setTemp(bool is_temp);

    iterator begin();
    iterator end();

protected:
    int key;
    bool is_temp;

};

/**
 *  Encapsulation of communication with Bohrium runtime.
 *  Implemented as a singleton.
 *
 *  Note: not thread-safe.
 */
class Runtime {
public:
    static Runtime* instance();

    ~Runtime();

    template <typename T>   // x = y + z
    void enqueue( bh_opcode opcode, Operand<T> & op0, Operand<T> & op1, Operand<T> & op2); 

    template <typename T>   // x = y + 1;
    void enqueue( bh_opcode opcode, Operand<T> & op0, Operand<T> & op1, T const& op2);    

    template <typename T>   // x = 1 + y;
    void enqueue( bh_opcode opcode, Operand<T> & op0, T const& op1, Operand<T> & op2);    

    template <typename T>   // x = y;
    void enqueue( bh_opcode opcode, Operand<T> & op0, Operand<T> & op1);                  

    template <typename T>   // x = 1.0;
    void enqueue( bh_opcode opcode, Operand<T> & op0, T const& op1);                     

    template <typename T>   // SYS: FREE, SYNC, DISCARD;
    void enqueue( bh_opcode opcode, Operand<T> & op0);

    bh_intp flush();

private:
    static Runtime* pInstance;                  // Singleton instance pointer.

    bh_instruction  queue[BH_CPP_QUEUE_MAX];    // Bytecode queue
    bh_intp         queue_size;

    bh_init         vem_init;                   // Bohrium interface
    bh_execute      vem_execute;
    bh_shutdown     vem_shutdown;
    bh_reg_func     vem_reg_func;

    bh_component    **components,               // Bohrium component setup
                    *self_component,
                    *vem_component;

    bh_intp children_count;

    Runtime();                                  // Ensure no external instantiation.

};

template <typename T>
class Vector : public Operand<T> {
public:

    Vector( Vector const& vector );
    Vector( int d0 );
    Vector( int d0, int d1 );

    //~Vector();

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
    Vector& operator=( T const& rhs );  // Used for initialization / assignment.
    Vector& operator=( Vector & rhs );

    Vector& operator[]( int index );    // This is a performance killer.

    Vector& operator++();               // Increment all elements in container
    Vector& operator++( int );
    Vector& operator--();               // Decrement all elements in container
    Vector& operator--( int );

    Vector& operator+=( const T rhs );  // Compound assignment operators
    Vector& operator+=( Vector & rhs );

    Vector& operator-=( const T rhs );
    Vector& operator-=( Vector & rhs );

    Vector& operator*=( const T rhs );
    Vector& operator*=( Vector & rhs );

    Vector& operator/=( const T rhs );
    Vector& operator/=( Vector & rhs );

    Vector& operator%=( const T rhs );
    Vector& operator%=( Vector & rhs );

    Vector& operator>>=( const T rhs );
    Vector& operator>>=( Vector & rhs );

    Vector& operator<<=( const T rhs );
    Vector& operator<<=( Vector & rhs );

    Vector& operator&=( const T rhs );
    Vector& operator&=( Vector & rhs );

    Vector& operator^=( const T rhs );
    Vector& operator^=( Vector & rhs );

    Vector& operator|=( const T rhs );
    Vector& operator|=( Vector & rhs );

};

}

#include "runtime.hpp"      // Communication with Bohrium runtime
#include "vector.hpp"       // Vector (De)Constructor.
#include "operators.hpp"    // Vector operations via operator-overloads.
#include "functions.hpp"    // Vector operations via functions.
#include "sugar.hpp"        // Pretty print functions and the like...

#endif
