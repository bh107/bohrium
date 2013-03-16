#include <stdexcept>
#include <sstream>
#include <iostream>

using namespace std;

#define MAX_RANK 10

class slice {
public:
    slice(int base, int stride, int bound) : base(base), stride(stride), bound(bound), dim(0), max(0) { }
    slice(int base, int stride, int bound, int dim, int max) : base(base), stride(stride), bound(bound), dim(dim), max(max) {
    
    }

    slice& operator[](slice& op);

    int base, stride, bound, dim, max;
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

    slice& operator[](slice& op);
    array<ValueType>& operator=(slice& op);

private:
    int shape[MAX_RANK];
    int dim;

};

template <typename ValueType>
slice& array<ValueType>::operator[](slice& op) {
    cout << dim-- << ":" << op.base << endl;
    return op;
}

template <typename ValueType>
array<ValueType>& array<ValueType>::operator=(slice& op) {
    cout << "Now assign it!" << op.base;
    cout << dim-- << ":" << op.base << endl;
    return *this;
}

slice& slice::operator[](slice& op) {
    cout << ":" << op.base << endl;
    return op;
}

inline
slice& cut(int base, int stride, int bound)
{
    return *(new slice(base, stride, bound));
}

int main() {

    array<double> x(9);
    array<double> y(9);

    y = x[cut(1,1,1)][cut(2,2,2)];

}
