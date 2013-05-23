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
#include <complex>
#include <list>

#define BH_CPP_QUEUE_MAX 1000
#include "iterator.hpp"
#include <stdexcept>

#ifdef DEBUG
#define DEBUG_PRINT(...) do{ fprintf( stderr, __VA_ARGS__ ); } while( false )
#else
#define DEBUG_PRINT(...) do{ } while ( false )
#endif

namespace bh {

const double PI_D = 3.141592653589793238462;
const float  PI_F = 3.14159265358979f;
const float  PI   = 3.14159265358979f;

template <typename T>   // Forward declaration
class multi_array;

int64_t unpack_shape(int64_t *shape, size_t index, size_t arg)
{
    shape[index] = arg;
    return 0;
}

template <typename ...Args>
int64_t unpack_shape(int64_t *shape, size_t index, size_t arg, Args... args)
{
    shape[index] = arg;
    unpack_shape(shape, ++index, args...);

    return 1;
}

//
// Extensions
//

enum reducible {
    ADD         = BH_ADD,
    MULTIPLY    = BH_MULTIPLY,
    MIN         = BH_MINIMUM,
    MAX         = BH_MAXIMUM,
    LOGICAL_AND = BH_LOGICAL_AND,
    LOGICAL_OR  = BH_LOGICAL_OR,
    BITWISE_AND = BH_BITWISE_AND,
    BITWISE_OR  = BH_BITWISE_OR
};

//
// Slicing
//
class slice_range {
public:
    slice_range();
    slice_range(int begin, int end, size_t stride);

    int begin, end;
    size_t stride;
};

template <typename T>
class slice {
public:
    slice(multi_array<T>& op);

    slice& operator[](int rhs);
    slice& operator[](slice_range& rhs);
    multi_array<T>& operator=(T rhs);

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
    // ** Constructors **
    multi_array();                              // Empty

    template <typename ...Dimensions>           // Variadic constructor
    multi_array(Dimensions... shape);

    multi_array(multi_array<T> const& operand); // Copy

    // ** Deconstructor **
    ~multi_array();

    // Types:
    typedef multi_array_iter<T> iterator;

    // Getter / Setter:
    size_t getKey() const;
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
   
    multi_array& operator=(const T& rhs);           // Initialization / assignment.
    multi_array& operator=(multi_array<T>& rhs);    // Initialization / assignment.

    template <typename In>
    multi_array<T>& operator=(multi_array<In>& rhs);    // Initialization / assignment.

    multi_array& operator=(slice<T>& rhs );         // Initialization / assignment.

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

    size_t len();
    multi_array<T>& copy();                 // Explicity create a copy of array
    multi_array<T>& flatten();              // Create a flat copy of the array

    multi_array<T>& update(multi_array& rhs); // Fill the array with values from another.

    multi_array<T>& transpose();
    
    template <typename Ret>                 // Typecast; implicit copy
    multi_array<Ret>& as();

    void link(size_t);
    size_t unlink();

protected:
    size_t key;
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
                            // Input and output are of the same type
                            
    template <typename T>   // SYS: FREE, SYNC, DISCARD;
    void enqueue(bh_opcode opcode, multi_array<T>& op0);

    template <typename T>   // x = y + z
    void enqueue(bh_opcode opcode, multi_array<T>& op0, multi_array<T> & op1, multi_array<T> & op2); 

    template <typename T>   // x = y + 1;
    void enqueue(bh_opcode opcode, multi_array<T>& op0, multi_array<T> & op1, const T& op2);

    template <typename T>   // x = 1 + y;
    void enqueue(bh_opcode opcode, multi_array<T>& op0, const T& op1, multi_array<T> & op2);

    template <typename T>   // x = y;
    void enqueue(bh_opcode opcode, multi_array<T>& op0, multi_array<T> & op1);                  

    template <typename T>   // x = 1.0;
    void enqueue(bh_opcode opcode, multi_array<T>& op0, const T& op1);

                                            // Same input but different output type
    template <typename Ret, typename In>    // x = y;
    void enqueue(bh_opcode opcode, multi_array<Ret>& op0, multi_array<In>& op1);

    template <typename Ret, typename In>    // x = y < z
    void enqueue(bh_opcode opcode, multi_array<Ret>& op0, multi_array<In>& op1, multi_array<In>& op2); 

    template <typename Ret, typename In>    // x = y < 1;
    void enqueue(bh_opcode opcode, multi_array<Ret>& op0, multi_array<In>& op1, const In& op2);    

    template <typename Ret, typename In>    // x = 1 < y;
    void enqueue(bh_opcode opcode, multi_array<Ret>& op0, const In& op1, multi_array<In>& op2);    

                                            // Mixed input, ret is same as first operand
    template <typename Ret, typename In>    // pow(...,2), reduce(..., 2)
    void enqueue(bh_opcode opcode, multi_array<Ret>& op0, multi_array<Ret>& op1, const In& op2);

    template <typename T>                   // Userfunc / extensions
    void enqueue(bh_userfunc* rinstr);

    size_t flush();
    size_t get_queue_size();

    template <typename T>
    multi_array<T>& op();

    template <typename T>
    multi_array<T>& temp();

    template <typename T, typename ...Dimensions>
    multi_array<T>& temp(Dimensions... shape);

    template <typename T>
    multi_array<T>& temp(multi_array<T>& input);

    template <typename T>
    multi_array<T>& view(multi_array<T>& base);

    template <typename T>
    multi_array<T>& temp_view(multi_array<T>& base);

    int64_t random_id;                          // Extension IDs

    void trash(size_t key);

private:

    size_t deallocate_meta(size_t count);       // De-allocate bh_arrays
    size_t deallocate_ext();                    // De-allocate user functions structs

    size_t execute();                           // Send instructions to Bohrium
    size_t guard();                             // Prevent overflow of instruction-queue

    static Runtime* pInstance;                  // Singleton instance pointer.

    bh_instruction  queue[BH_CPP_QUEUE_MAX];    // Bytecode queue
    bh_userfunc     *ext_queue[BH_CPP_QUEUE_MAX];
    size_t          ext_in_queue;
    size_t          queue_size;

    bh_init         vem_init;                   // Bohrium interface
    bh_execute      vem_execute;
    bh_shutdown     vem_shutdown;
    bh_reg_func     vem_reg_func;

    bh_component    **components,               // Bohrium component setup
                    *self_component,
                    *vem_component;

    int64_t children_count;

    std::list<size_t> garbage;

    Runtime();                                  // Ensure no external instantiation.

};

template <typename T>       // Generators / Initializers
multi_array<T>& empty(size_t n, ...);

template <typename T>
multi_array<T>& zeros(size_t n, ...);

template <typename T>
multi_array<T>& ones(size_t n, ...);

template <typename T>
multi_array<T>& random(size_t n, ...);

template <typename T>
multi_array<T>& arange();

                            // REDUCTIONS
template <typename T>       // Partial
multi_array<T>& reduce(multi_array<T>& op, reducible opc, size_t axis);
                            // FULL
template <typename T>       // Numeric 
multi_array<T>& sum(multi_array<T>& op);      

template <typename T>
multi_array<T>& product(multi_array<T>& op);

template <typename T>
multi_array<T>& min(multi_array<T>& op);

template <typename T>       // Boolean
multi_array<T>& max(multi_array<T>& op);

template <typename T>
multi_array<bool>& any(multi_array<T>& op);

template <typename T>
multi_array<bool>& all(multi_array<T>& op);

template <typename T>       // Mixed...
multi_array<size_t>& count(multi_array<T>& op);

template <typename T>       // Turn the result of full reduction into a scalar
T scalar(multi_array<T>& op);

template <typename T>
void pprint(multi_array<T>& op);
}

#include "multi_array.hpp"  // Operand definition.
#include "broadcast.hpp"    // Operand manipulations.
#include "slicing.hpp"      // Operand slicing / explicit views / aliases
#include "runtime.hpp"      // Communication with Bohrium runtime
#include "reduction.hpp"    // Communication with Bohrium runtime
#include "generator.hpp"    // Communication with Bohrium runtime

#include "operators.hpp"    // DSEL Operations via operator-overloads.
#include "functions.hpp"    // DSEL Operations via functions.
#include "sugar.hpp"        // DSEL Additional sugar... 

#endif
