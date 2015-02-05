#include <bxx/bohrium.hpp>
#include <iostream>

using namespace std;
using namespace bxx;

int main() {
    multi_array<float> a = ones<float>(3);
    cout << "HW " << a << endl;
    return 0;
}
