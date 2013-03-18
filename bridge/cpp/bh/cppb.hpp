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
#include <stdexcept>
#include <array>

#ifdef DEBUG
#define DEBUG_PRINT(...) do{ fprintf( stderr, __VA_ARGS__ ); } while( false )
#else
#define DEBUG_PRINT(...) do{ } while ( false )
#endif

namespace bh {

class slice_range {
public:
    slice_range() : begin(0), end(-1), stride(1) {}
    slice_range(int begin, int end, unsigned int stride) : begin(begin), end(end), stride(stride) {}

    int begin, end;
    unsigned int stride;
};

enum slice_bound { ALL, FIRST, LAST };

template <typename T>
class multi_array;

template <typename T>
class slice {
public:
    slice(multi_array<T>& op) : op(&op), dims(0) {}

    slice& operator[](int rhs)
    {
        std::cout << "slice[int] [dim=" << dims << "] " << rhs <<std::endl;
        ranges[dims].begin = rhs;
        ranges[dims].end   = rhs;
        dims++;
        return *this;
    }

    slice& operator[](slice_bound rhs)
    {
        std::cout << "slice[ALL] [dim=" << dims << "] " << rhs <<std::endl;
        dims++;
        return *this;
    }

    slice& operator[](slice_range& rhs)
    {
        std::cout << "slice[range] [dim=" << dims << "]" <<std::endl;
        ranges[dims] = rhs;
        dims++;
        return *this;
    }

    // Create a actual view of the slice.
    bh::multi_array<T> view()
    {
        std::cout << " Create the view! " << dims <<std::endl;
        for(int i=0; i<dims; ++i ) {
            std::cout << "[Dim="<< i << "; " << ranges[i].begin << "," \
                                        << ranges[i].end << "," \
                                        << ranges[i].stride << "]" \
                                        <<std::endl;
            //std::cout << "PUKE" <<std::endl;
        }
        return NULL;
    }

private:
    multi_array<T>* op;             // The op getting sliced

    int dims;                               // The amount of dims covered by the slice
    std::array<slice_range, BH_MAXDIM> ranges;    // The ranges...

};

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
    slice<T>& operator[](slice_bound rhs);          // Select the entire dimension
    slice<T>& operator[](slice_range& rhs);         // Select a range (begin, end, stride)

    multi_array& operator=( T const& rhs );         // Initialization / assignment.
    multi_array& operator=( multi_array & rhs );

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

    multi_array& operator++();               // Increment all elements in container
    multi_array& operator++(int);
    multi_array& operator--();               // Decrement all elements in container
    multi_array& operator--(int);

protected:
    unsigned int key;
    bool temp;

private:
    void init();

};





inline
slice_range& _(int base, int end, unsigned int stride)
{
    return *(new slice_range(base, stride, end));
}

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

    template <typename T>
    multi_array<T>& temp_view(multi_array<T>& base);

    bh_intp queued() { return queue_size; }

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
#include "broadcast.hpp"    // Operand manipulations.
#include "slicing.hpp"      // Operand slicing / explicit views / aliases
#include "runtime.hpp"      // Communication with Bohrium runtime

#include "operators.hpp"    // DSEL Operations via operator-overloads.
#include "functions.hpp"    // DSEL Operations via functions.
#include "sugar.hpp"        // DSEL Additional sugar... 

#endif
