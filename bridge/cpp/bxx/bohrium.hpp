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
#include <map>

#include <bh.h>
#include "iterator.hpp"

#ifdef DEBUG
#define DEBUG_PRINT(...) do{ fprintf( stderr, __VA_ARGS__ ); } while( false )
#else
#define DEBUG_PRINT(...) do{ } while ( false )
#endif

namespace bxx {

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
    LOGICAL_XOR = BH_LOGICAL_XOR,
    BITWISE_AND = BH_BITWISE_AND,
    BITWISE_OR  = BH_BITWISE_OR,
    BITWISE_XOR  = BH_BITWISE_XOR
};

enum scannable {
    SUM     = BH_ADD_ACCUMULATE,
    PRODUCT = BH_MULTIPLY_ACCUMULATE
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
    bxx::multi_array<T>& view();

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
    multi_array(const uint64_t rank, const int64_t* shapes);
    multi_array(bh_base* base, uint64_t rank, const int64_t start, const int64_t* shape, const int64_t* stride);

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

    multi_array& operator()(const void* data);      // Update / copy from a void pointer

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
    void link(bh_base* base);               // Bohrium Runtime Specifics
    bh_base* unlink();

    bh_base* getBase() const;

    void* getBaseData(void);                // These are NOT for the faint of heart!
    void setBaseData(void* data);           // These are only intended for the C-bridge.

    bool getTemp() const;
    void setTemp(bool temp);
    bool linked() const;
    bool initialized() const;               // Determine if the array is initialized (has a bh_base)
    bool allocated() const;                 // Determine if the array is intitialized and data for it is allocated
    void sync();

protected:
    bool temp_;

private:
    void reset_meta();						// Helper, shared among constructors

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

    ~Runtime();         // Deconstructor

    //
    //  Lazy evaluation through instruction queue
    //
    template <typename Out, typename In1, typename In2>
    void enqueue(bh_opcode opcode, multi_array<Out>& op0, multi_array<In1>& op1, multi_array<In2>& op2);
    
    template <typename Out, typename In1, typename In2>
    void enqueue(bh_opcode opcode, multi_array<Out>& op0, multi_array<In1>& op1, const In2 op2);
    
    template <typename Out, typename In1, typename In2>
    void enqueue(bh_opcode opcode, multi_array<Out>& op0, const In1 op1, multi_array<In2>& op2);

    template <typename Out, typename In>
    void enqueue(bh_opcode opcode, multi_array<Out>& op0, multi_array<In>& op1);

    template <typename Out, typename In>
    void enqueue(bh_opcode opcode, multi_array<Out>& op0, const In op1);

    template <typename Out>
    void enqueue(bh_opcode opcode, multi_array<Out>& op0);

    template <typename Out>
    void enqueue(bh_opcode opcode, multi_array<Out>& op0, const uint64_t op1, const uint64_t op2);

    template <typename Ret, typename In1, typename In2>
    void enqueue_extension(const std::string& name, multi_array<Ret>& op0, multi_array<In1>& op2, multi_array<In2>& op3);

    size_t flush();
    size_t get_queue_size();

    //
    //  Typechecker
    //
    template <size_t Opcode, typename Out, typename In1, typename In2>
    void typecheck(void);

    template <size_t Opcode, typename Out, typename In1>
    void typecheck(void);

    template <size_t Opcode, typename Out>
    void typecheck(void);   

    //
    //  Operand construction
    //
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

    void trash(bh_base* base);

    std::map<bh_base*, size_t> ref_count;       // Count references to bh_base
    std::map<bh_base*, size_t> ext_allocated;   // Lookup register for externally allocated data

private:
                                                // Bohrium
    bh_component        bridge;
    bh_component_iface  *runtime;

    std::map<std::string, bh_opcode> extensions;// Register of extensions
    size_t extension_count;


    bh_instruction  queue[BH_CPP_QUEUE_MAX];    // Bytecode queue
    size_t          ext_in_queue;
    size_t          queue_size;

    std::list<bh_base*> garbage;                // Collection of bh_base which will
                                                // be deleted when the current batch is flushed.

    Runtime();                                  // Ensure no external instantiation.
    Runtime(Runtime const&);
    void operator=(Runtime const&);

    size_t deallocate_meta(size_t count);       // De-allocate bh_base
    size_t deallocate_ext();                    // De-allocate userdefined functions structs

    size_t execute();                           // Send instructions to Bohrium
    size_t guard();                             // Prevent overflow of instruction-queue

};

}

#include "multi_array.hpp"  // Operand definition.
#include "slicing.hpp"      // Operand slicing / explicit views / aliases

#include "runtime.hpp"              // Communication with Bohrium runtime
#include "runtime.broadcast.hpp"    // Operand broadcasting
#include "runtime.typechecker.hpp"  // Array operations - typechecker
#include "runtime.operations.hpp"   // Array operations

#include "reduction.hpp"    // DSEL Reduction
#include "scan.hpp"         // DSEL Scan operation
#include "generator.hpp"    // DSEL Generators 

#include "operators.hpp"    // DSEL Operations via operator-overloads.
#include "sugar.hpp"        // DSEL Additional sugar...

#endif
