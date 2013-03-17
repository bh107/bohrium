#include <stdexcept>
#include <sstream>
#include <iostream>
#include <map>

using namespace std;

#define MAX_RANK 10

template <typename ValueType>
class array;

enum slice_bound { ALL, FIRST, LAST, NONE };
class slice_range {
public:
    slice_range() : begin(0), end(0), stride(0) {}
    slice_range(int begin, int end, int stride) : begin(begin), end(end), stride(stride) {}
    int begin, end, stride;
};

template <typename ValueType>
class slice {
public:
    // Constructors
    slice(int dim)                      : dim(dim), op(NULL) {
    }
    slice(int dim, slice_range range)   : dim(dim), op(NULL) {
        ranges[dim] = range;
    }

    slice& operator[](slice_range& rhs);
    slice& operator[](slice_bound rhs);

    int dim;
    map<int,slice_range> ranges;
    array<ValueType>* op;
};


template <typename ValueType>
class array {
public:
    
    array(int n) : dim(1) {
        for(int i=0; i<MAX_RANK;i++) {
            shape[0] = 0;
        }
        shape[MAX_RANK-1] = n;
    }

    slice<ValueType>& operator[](slice_range& rhs);
    slice<ValueType>& operator[](slice_bound rhs);

    array<ValueType>& operator=(slice<ValueType>& rhs);

private:
    int shape[MAX_RANK];
    int dim;

};

// ASSIGN
template <typename ValueType>
array<ValueType>& array<ValueType>::operator=(slice<ValueType>& rhs) {
    cout << "RESOLVE=" << rhs.dim << "." << rhs.ranges.size() << endl;

    std::map<int, slice_range>::iterator i = rhs.ranges.begin();
    for( ; i != rhs.ranges.end(); ++i )
    {
        // i->first is your key
        cout << i->first << endl;
        cout << i->second.begin << endl;
        // i->second is it''s value

    }

    return *this;
}

// SLICE-ARRAY
template <typename ValueType>
slice<ValueType>& array<ValueType>::operator[](slice_range& rhs) {
    cout << "array[range] [dim=" << 0 << "]" << endl;
    return *(new slice<ValueType>(0, rhs));
}

template <typename ValueType>
slice<ValueType>& array<ValueType>::operator[](slice_bound rhs) {
    cout << "array[ALL] [dim=" << 0 << "] " << rhs << endl;
    return *(new slice<ValueType>(0));
}

// SLICE
template <typename ValueType>
slice<ValueType>& slice<ValueType>::operator[](slice_range& rhs) {
    dim++;
    cout << "slice[range] [dim=" << dim << "]" << endl;
    ranges.insert(pair<int, slice_range>(dim, rhs));
    return *this;
}

template <typename ValueType>
slice<ValueType>& slice<ValueType>::operator[](slice_bound rhs) {
    dim++;
    cout << "slice[ALL] [dim=" << dim << "] " << rhs << endl;
    return *this;
}

inline
slice_range& cut(int base, int end, int stride)
{
    return *(new slice_range(base, stride, end));
}

int main() {

    array<double> x(9);
    array<double> y(9);

    //y = x[cut(1,1,1)][cut(2,2,2)];
    y = x[cut(10,100,1)][cut(20,50,2)];
    //y = x[ALL][ALL];

}
