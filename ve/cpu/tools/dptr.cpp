#include <bxx/bohrium.hpp>
#include <iostream>

using namespace std;
using namespace bxx;

int main() {
    void* davs;

    multi_array<float> a, b, c;
    a = ones<float>(3);

    cout << "h1 " << a << endl;
    davs = a.getBaseData();
    cout << "H1 " << a.getBaseData() << endl;

    a.setBaseData(NULL);
    cout << "H2 " << a.getBaseData() << endl;

    a.setBaseData(davs);
    cout << "H2 " << a.getBaseData() << endl;

    return 0;
}
