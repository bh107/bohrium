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
#ifndef __BOHRIUM_BRIDGE_CPP
#define __BOHRIUM_BRIDGE_CPP

#define BH_CPP_QUEUE_MAX 1000

#include <stdexcept>
#include <complex>
#include <list>

#include "bh.h"
#include "iterator.hpp"

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

inline int64_t unpack_shape(int64_t *shape, size_t index, size_t arg)
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

#ifndef END
#define END 0
#endif

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
    bool inclusive_end;
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
    bh_view meta;

    // ** Constructors **
    multi_array();                              // Empty
    multi_array(const multi_array<T> &operand); // Copy

    template <typename OtherT>
    multi_array(const multi_array<OtherT> &operand);    // Copy

    template <typename ...Dimensions>                   // Variadic constructor
    multi_array(Dimensions... dims);

    // ** Deconstructor **
    ~multi_array();

    // Types:
    typedef multi_array_iter<T> iterator;

    size_t len();
    int64_t shape(int64_t dim);             // Probe for the shape of the given dimension
    unsigned long getRank() const;

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

    multi_array& operator()(const T& n);            // Update
    multi_array& operator()(multi_array<T>& rhs);
   
    multi_array& operator=(const T& rhs);           // Initialization / assignment.
    multi_array& operator=(multi_array<T>& rhs);    // Initialization / assignment.

    template <typename In>
    multi_array<T>& operator=(multi_array<In>& rhs);// Initialization / assignment.

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

    void link();                            // Bohrium Runtime Specifics
    void link(bh_base *base_ptr);     // Bohrium Runtime Specifics
    bh_base* unlink();

    bh_base* getBase() const;
    bool getTemp() const;
    void setTemp(bool temp);
    bool linked() const;
    bool initialized() const;

protected:
    bool temp;
    bh_base *base;

private:

};

/**
 *  Encapsulation of communication with Bohrium runtime.
 *  Implemented as a Singleton.
 *
 *  Note: Not thread-safe.
 */
class Runtime {
public:
    static Runtime& instance(); // Singleton method
    ~Runtime();                 // Deconstructor

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

    template <typename T, typename OtherT>
    multi_array<T>& temp(multi_array<OtherT>& input);

    template <typename T>
    multi_array<T>& view(multi_array<T>& base);

    template <typename T>
    multi_array<T>& temp_view(multi_array<T>& base);

    int64_t random_id;                          // Extension IDs

    void trash(bh_base *base_ptr);

private:
                                                // Bohrium
    bh_component    *bridge;
    bh_component    *runtime;

    bh_instruction  queue[BH_CPP_QUEUE_MAX];    // Bytecode queue
    bh_userfunc     *ext_queue[BH_CPP_QUEUE_MAX];
    size_t          ext_in_queue;
    size_t          queue_size;
                                                // DSEL stuff
    std::list<bh_base*> garbage;                // NOTE: This is probably deprecated with bh_base...
                                                // Collection of bh_base which will
                                                // be deleted when the current batch is flushed.

    Runtime();                                  // Ensure no external instantiation.
    Runtime(Runtime const&);
    void operator=(Runtime const&);

    size_t deallocate_meta(size_t count);       // De-allocate bh_base
    size_t deallocate_ext();                    // De-allocate userdefined functions structs

    size_t execute();                           // Send instructions to Bohrium
    size_t guard();                             // Prevent overflow of instruction-queue

};

template <typename T>       // Generators / Initializers
multi_array<T>& value(T val, size_t n, ...);

template <typename T>       
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

template <typename T, typename FromT>     // Typecast; implicit copy
multi_array<T>& as(multi_array<FromT>& rhs);

template <typename T, typename ...Dimensions>   // 
multi_array<T>& view_as(multi_array<T>& rhs, Dimensions... shape);

                            //
                            // What are these called? Transformers??? :)
                            //
template <typename T>
multi_array<T>& copy(multi_array<T>& rhs);     // Explicity create a copy of array

template <typename T>
multi_array<T>& flatten(multi_array<T>& rhs);  // Create a flat copy of the array

template <typename T>
multi_array<T>& transpose(multi_array<T>& rhs);

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
