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

#define BH_CPP_QUEUE_MAX 512
#include "iterator.hpp"
#include <stdexcept>
#include <vector>

#ifdef DEBUG
#define DEBUG_PRINT(...) do{ fprintf( stderr, __VA_ARGS__ ); } while( false )
#else
#define DEBUG_PRINT(...) do{ } while ( false )
#endif

namespace bh {

template <typename T>   // Forward declaration
class multi_array;

//
// Extensions
//
static bh_intp reduce_id;
static bh_intp random_id;

enum reducible {
    ADD         = BH_ADD,
    SUBTRACT    = BH_SUBTRACT,
    MULTIPLY    = BH_MULTIPLY
};

//
// Slicing
//
class slice_range {
public:
    slice_range();
    slice_range(int begin, int end, unsigned int stride);

    int begin, end;
    unsigned int stride;
};

template <typename T>
class slice {
public:
    slice(multi_array<T>& op);

    slice& operator[](int rhs);
    slice& operator[](slice_range& rhs);

    // Create a actual view of the slice
    bh::multi_array<T>& view();

private:
    multi_array<T>* op;                 // The op getting sliced

    int dims;                           // The amount of dims covered by the slice
    slice_range ranges[BH_MAXDIM];      // The ranges...

};

//
// The Abstraction
//
template <typename T>
class multi_array {
public:
    // Constructors:
    multi_array();
    multi_array( unsigned int n );
    multi_array( unsigned int m, unsigned int n );
    multi_array( unsigned int d2, unsigned int d1, unsigned int d0 );
    multi_array( multi_array<T> const& operand );

    // Deconstructor:
    ~multi_array();

    // Types:
    typedef multi_array_iter<T> iterator;

    // Getter / Setter:
    unsigned int getKey() const;
    unsigned long getRank() const;
    bool getTemp() const;
    void setTemp(bool temp);

    // Iterator
    iterator begin();
    iterator end();

    //
    // Operators: 
    //
    // =, [], (), -> must be "internal" (nonstatic member functions) and thus declared here.
    //
    // Definitions are provided in:
    //
    // - multi_array.hpp for those implemented by hand ([], ++, --, ostream<< ).
    // - slicing.hpp: Auxilary behavior of the [] operator.
    // - operators.hpp: defined code-generator.
    //
                                                    // Slicing / explicit view
    slice<T>& operator[](int rhs);                  // Select a single element / dimension
    slice<T>& operator[](slice_range& rhs);         // Select a range (begin, end, stride)

    multi_array& operator()(const T& n);              // Shaping / reshaping
    multi_array& operator()(const T& m, const T& n);              // Shaping / reshaping
    multi_array& operator()(const T& d2, const T& d1, const T& d0);              // Shaping / reshaping
   
    multi_array& operator=(const T& rhs);         // Initialization / assignment.
    multi_array& operator=(multi_array<T>& rhs);         // Initialization / assignment.
    multi_array& operator=(slice<T>& rhs );         // Initialization / assignment.

    /* 
    multi_array& operator=(multi_array & rhs );
    */
    multi_array& operator+=(const T& rhs);          // Compound assignment / increment
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

    multi_array& operator++();              // Increment all elements in container
    multi_array& operator++(int);
    multi_array& operator--();              // Decrement all elements in container
    multi_array& operator--(int);

    multi_array<T>& copy();                 // Explicity create a copy of array
    multi_array<T>& flatten();              // Create a flat copy of the array

                                            // Extensions
    multi_array<T>& reduce(reducible op, int axis);

protected:
    unsigned int key;
    bool temp;

private:
    void init();

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
    void enqueue(bh_opcode opcode, multi_array<T> & op0, T const& op1);                     
    template <typename T>   // SYS: FREE, SYNC, DISCARD;
    void enqueue(bh_opcode opcode, multi_array<T> & op0);

    template <typename T>
    void enqueue(bh_userfunc* rinstr);

    bh_intp flush();

    template <typename T>
    multi_array<T>& op();

    template <typename T>
    multi_array<T>& temp();

    template <typename T>
    multi_array<T>& temp(multi_array<T>& input);

    template <typename T>
    multi_array<T>& view(multi_array<T>& base);

    template <typename T>
    multi_array<T>& temp_view(multi_array<T>& base);

    bh_intp guard();

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

template <typename T>       // These should be "generators"...
multi_array<T>& empty();

template <typename T>
multi_array<T>& zeros();

template <typename T>
multi_array<T>& ones();

template <typename T>
multi_array<T>& arange();

template <typename T>
multi_array<T>& random();

template <typename T>
multi_array<T>& random(int n);

}

#include "multi_array.hpp"  // Operand definition.
#include "broadcast.hpp"    // Operand manipulations.
#include "slicing.hpp"      // Operand slicing / explicit views / aliases
#include "runtime.hpp"      // Communication with Bohrium runtime
#include "extensions.hpp"   // Communication with Bohrium runtime

#include "operators.hpp"    // DSEL Operations via operator-overloads.
#include "functions.hpp"    // DSEL Operations via functions.
#include "sugar.hpp"        // DSEL Additional sugar... 

#endif
