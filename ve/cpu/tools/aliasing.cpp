#include <bxx/bohrium.hpp>
#include <iostream>

using namespace std;
using namespace bxx;

int main() {
    multi_array<float> a, b, c;
    cout << "1" << endl;
    a = ones<float>(3);
    cout << "2" << endl;
    b = a;
    cout << "3" << endl;
    a = ones<float>(6);
    cout << "4" << endl;
    cout << "HW " << a << endl;
    cout << "HW " << b << endl;
    //cout << "HW " << b << endl;
    return 0;
}
