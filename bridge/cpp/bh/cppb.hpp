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
class multi_array {
public:
    multi_array();
    multi_array( unsigned int n );
    multi_array( unsigned int m, unsigned int n );
    multi_array( unsigned int d2, unsigned int d1, unsigned int d0 );
    multi_array( multi_array<T> const& operand );
    ~multi_array();

    // Types:
    typedef multi_array_iter<T> iterator;

    unsigned int getKey() const;
    bool isTemp() const;
    void setTemp(bool is_temp);

    iterator begin();
    iterator end();

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

    multi_array& operator[]( int index );    // This is a performance killer.

    multi_array& operator=( T const& rhs );  // Used for initialization / assignment.
    multi_array& operator=( multi_array & rhs );

    multi_array& operator+=(const T& rhs);   // Compound assignment operators / increment
    multi_array& operator+=(multi_array& rhs);

    multi_array& operator-=(const T& rhs);
    multi_array& operator-=(multi_array& rhs);

    multi_array& operator*=(const T& rhs);
    multi_array& operator*=(multi_array& rhs);

    multi_array& operator/=(const T& rhs);
    multi_array& operator/=(multi_array &rhs);

    multi_array& operator%=(const T& rhs);
    multi_array& operator%=(multi_array &rhs);

    multi_array& operator>>=(const T& rhs);
    multi_array& operator>>=(multi_array& rhs);

    multi_array& operator<<=(const T& rhs);
    multi_array& operator<<=(multi_array& rhs);

    multi_array& operator&=(const T& rhs);
    multi_array& operator&=(multi_array& rhs);

    multi_array& operator^=(const T& rhs);
    multi_array& operator^=(multi_array& rhs);

    multi_array& operator|=(const T& rhs);
    multi_array& operator|=(multi_array& rhs);

    multi_array& operator++();               // Increment all elements in container
    multi_array& operator++(int);
    multi_array& operator--();               // Decrement all elements in container
    multi_array& operator--(int);

protected:
    unsigned int key;
    bool is_temp;

private:
    void init();

};

template <typename T>
class vector : public multi_array<T> {
public:
    vector(unsigned int n);
    
    using multi_array<T>::operator=;

};

template <typename T>
class matrix : public multi_array<T> {
public:
    matrix(unsigned int m, unsigned int n);

    using multi_array<T>::operator=;
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
    void enqueue( bh_opcode opcode, multi_array<T> & op0, multi_array<T> & op1, multi_array<T> & op2); 

    template <typename T>   // x = y + 1;
    void enqueue( bh_opcode opcode, multi_array<T> & op0, multi_array<T> & op1, T const& op2);    

    template <typename T>   // x = 1 + y;
    void enqueue( bh_opcode opcode, multi_array<T> & op0, T const& op1, multi_array<T> & op2);    

    template <typename T>   // x = y;
    void enqueue( bh_opcode opcode, multi_array<T> & op0, multi_array<T> & op1);                  

    template <typename T>   // x = 1.0;
    void enqueue( bh_opcode opcode, multi_array<T> & op0, T const& op1);                     

    template <typename T>   // SYS: FREE, SYNC, DISCARD;
    void enqueue( bh_opcode opcode, multi_array<T> & op0);

    bh_intp flush();

    template <typename T>
    multi_array<T>& op();

    template <typename T>
    multi_array<T>& temp();

    template <typename T>
    multi_array<T>& temp(multi_array<T>& input);

    template <typename T>
    multi_array<T>& view(multi_array<T>& base);

private:

    bh_intp guard();
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

}

#include "multi_array.hpp"  // Operand definition.
#include "broadcast.hpp"  // Operand definition.
#include "runtime.hpp"      // Communication with Bohrium runtime

#include "operators.hpp"    // DSEL Operations via operator-overloads.
#include "functions.hpp"    // DSEL Operations via functions.
#include "vector.hpp"       // DSEL Vector (De)Constructor.
#include "matrix.hpp"       // DSEL Matrix (De)Constructor.
#include "sugar.hpp"        // DSEL Sugar: pretty print functions and the like...

#endif
