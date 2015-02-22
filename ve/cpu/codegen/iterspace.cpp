#include <sstream>
#include <string>
#include "codegen.hpp"

using namespace std;
using namespace bohrium::core;

namespace bohrium{
namespace engine{
namespace cpu{
namespace codegen{

Iterspace::Iterspace(void) : iterspace_(NULL) {}
Iterspace::Iterspace(iterspace_t& iterspace) : iterspace_(&iterspace) {}

string Iterspace::name(void)
{
    stringstream ss;
    ss << "iterspace";
    return ss.str();
}

string Iterspace::ndim(void)
{
    stringstream ss;
    ss << _access_ptr(name(), "ndim");
    return ss.str();
}

string Iterspace::shape(uint32_t dim)
{
    stringstream ss;
    ss << _index(_access_ptr(name(), "shape"), dim);
    return ss.str();
}

}}}}
