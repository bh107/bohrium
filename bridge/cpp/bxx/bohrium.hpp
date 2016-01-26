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

#define is_aligned(POINTER, BYTE_COUNT) \
    (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0)

namespace bxx {

const double PI_D = 3.141592653589793238462;
const float  PI_F = 3.14159265358979f;
const float  PI   = 3.14159265358979f;

struct Export {
    enum Option {
        NONE        = 0x00,
        RELEASED    = 0x01,
        WO_ALLOC    = 0x02,
        WO_ZEROING  = 0x04
    };
};

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

inline int64_t nelements_shape(size_t arg)
{
    return arg;
}

template <typename ...Args>
int64_t nelements_shape(size_t arg, Args... args)
{
    return arg*nelements_shape(args...);
}

//
// Extensions
//

#ifndef END
#define END 0
#endif

enum scannable {
    SUM     = BH_ADD_ACCUMULATE,
    PRODUCT = BH_MULTIPLY_ACCUMULATE
};

//
// Slicing
//
class Slice {
public:
    Slice();
    Slice(int begin, int end, size_t step);

    int begin;
    int end;
    size_t step;
};

inline Slice::Slice() : begin(0), end(-1), step(1) {}

inline Slice::Slice(int begin, int end, size_t step)
    : begin(begin), end(end), step(step) {}

inline Slice _(int begin, int end, size_t step)
{
    return Slice(begin, end, step);
}

inline Slice _(int begin, int end)
{
    return _(begin, end, 1);
}

inline Slice _SI(int begin, int end, size_t step)
{
    return _(begin, end, step);
}

inline Slice _SI(int begin, int end)
{
    return _(begin, end, 1);
}

inline Slice _SE(int begin, int end, size_t step)
{
    return _(begin, end-1, step);
}

inline Slice _SE(int begin, int end)
{
    return _(begin, end-1, 1);
}

inline Slice _ALL(void)
{
    return _(0, -1, 1);
}

inline Slice _ABF(void)
{
    return _(1, -1, 1);
}

inline Slice _ABL(void)
{
    return _(0, -2, 1);
}

inline Slice _INNER(void)
{
    return _(1, -2, 1);
}

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
    // - operators.hpp: defined code-generator.
    //
                                                    // Slicing / explicit view
    multi_array<T>& operator[](int rhs);            // Select a single element / dimension
    multi_array<T>& operator[](Slice rhs);          // Select a range (begin, end, step)

    multi_array& operator()(const T& n);            // Update
    multi_array& operator()(multi_array<T>& rhs);

    //
    // Data movement
    //
    multi_array& operator()(const void* data);      // Update / copy from a void pointer

    T* data_export(void);
    void data_import(T* data);

    multi_array& operator=(const T& rhs);           // Initialization / assignment.
    multi_array& operator=(multi_array<T>& rhs);    // Initialization / assignment.

    template <typename In>
    multi_array<T>& operator=(multi_array<In>& rhs);// Initialization / assignment.

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

    int getSliceDim(void);
    void setSliceDim(int dim);

protected:
    bool temp_;
    int slicing_dim_;

private:
    void reset_meta();						// Helper, shared among constructors

};

template <typename TL, typename TR>
inline
bool identical(multi_array<TL>& left, multi_array<TR>& right)
{
    return static_cast<void*>(&left) == static_cast<void*>(&right);
}

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
    template <typename TO, typename TL, typename TR>
    void enqueue(bh_opcode opcode, multi_array<TO>& op0, multi_array<TL>& op1, multi_array<TR>& op2);
    
    template <typename TO, typename TL, typename TR>
    void enqueue(bh_opcode opcode, multi_array<TO>& op0, multi_array<TL>& op1, const TR op2);

    template <typename TO, typename TL, typename TR>
    void enqueue(bh_opcode opcode, multi_array<TO>& op0, const TL op1, multi_array<TR>& op2);

    template <typename TO, typename TI>
    void enqueue(bh_opcode opcode, multi_array<TO>& op0, multi_array<TI>& op1);

    template <typename TO, typename TI>
    void enqueue(bh_opcode opcode, multi_array<TO>& op0, const TI op1);

    template <typename TO>
    void enqueue(bh_opcode opcode, multi_array<TO>& op0);

    void enqueue(bh_opcode opcode);

    template <typename TO>
    void enqueue(bh_opcode opcode, multi_array<TO>& op0, const uint64_t op1, const uint64_t op2);

    template <typename T1, typename T2, typename T3>
    void enqueue_extension(const std::string& name, multi_array<T1>& op0, multi_array<T2>& op2, multi_array<T3>& op3);

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
    multi_array<T>& create(void);

    /**
        Construct a new "linked" array, that it, it has an
        associated `bh_base`.
        Shape is inherited by `input`.
    **/
    template <typename T, typename OtherT>
    multi_array<T>& create_base(multi_array<OtherT>& input);

    template <typename T>
    multi_array<T>& temp();

    template <typename T, typename OtherT>
    multi_array<T>& temp(multi_array<OtherT>& input);

    template <typename T>
    multi_array<T>& view(multi_array<T>& base);

    template <typename T>
    multi_array<T>& temp_view(multi_array<T>& base);

    void trash(bh_base* base);

    uint64_t getRandSeed(void);                 // Get/set the global random seed and state
    void setRandSeed(uint64_t);

    uint64_t getRandState(void);                
    void setRandState(uint64_t);

    std::map<bh_base*, size_t> ref_count;       // Count references to bh_base
    std::map<bh_base*, size_t> ext_allocated;   // Lookup register for externally allocated data

private:
    
    uint64_t    global_random_seed_;            // Random state and seed
    uint64_t    global_random_state_;           // TODO: Should be encapsulated in a "sugar-layer"

    bh_component        bridge;                 // Bohrium
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

#include "multi_array.hpp"          // Operand definition.

#include "runtime.hpp"              // Communication with Bohrium runtime
#include "runtime.broadcast.hpp"    // Operand broadcasting
#include "runtime.typechecker.hpp"  // Array operations - typechecker
#include "runtime.operations.hpp"   // Array operations
#include "runtime.extensions.hpp"   // Extensions

#include "reduction.hpp"    // DSEL Reduction
#include "scan.hpp"         // DSEL Scan operation
#include "generator.hpp"    // DSEL Generators 
#include "mapping.hpp"      // DSEL Gather / Scatter
#include "grids.hpp"        // DSEL Grid constructors
#include "movement.hpp"     // DSEL Export / import data from arrays
#include "visuals.hpp"      // DSEL Visualization

#include "operators.hpp"    // DSEL Operations via operator-overloads.
#include "sugar.hpp"        // DSEL Additional sugar...

#endif
