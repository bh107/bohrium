#include <sstream>
#include <string>
#include "codegen.hpp"

using namespace std;
using namespace bohrium::core;

namespace bohrium{
namespace engine{
namespace cpu{
namespace codegen{

// Element types;
string _int8(void)
{
    return "int8_t";
}

string _int16(void)
{
    return "int16_t";
}

string _int32(void)
{
    return "int32_t";
}

string _int64(void)
{
    return "int64_t";
}

string _uint8(void)
{
    return "uint8_t";
}

string _uint16(void)
{
    return "uint16_t";
}

string _uint32(void)
{
    return "uint32_t";
}

string _uint64(void)
{
    return "uint64_t";
}

string _float(void)
{
    return "float";
}
string _double(void)
{
    return "double";
}
string _float_complex(void)
{
    return "float complex";
}
string _double_complex(void)
{
    return "double complex";
}

string _ref(string object)
{
    stringstream ss;
    ss << "&(" << object << ")";
    return ss.str();
}

string _deref(string object)
{
    stringstream ss;
    ss << "*(" << object << ")";
    return ss.str();
}

string _index(string object, int64_t idx)
{
    stringstream ss;
    ss << object << "[" << idx << "]";
    return ss.str();
}

string _access(string object, string member)
{
    stringstream ss;
    ss << object << "." << member;
    return ss.str();
}

string _access_ptr(string object, string member)
{
    stringstream ss;
    ss << object << "->" << member;
    return ss.str();
}

string _ptr(string object)
{
    stringstream ss;
    ss << object << "*";
    return ss.str();
}

string _ptr_const(string object)
{
    stringstream ss;
    ss << object << "* const";
    return ss.str();
}

string _const(string object)
{
    stringstream ss;
    ss << "const " << object;
    return ss.str();
}

string _const_ptr(string object)
{
    stringstream ss;
    ss << "const " << object << "*";
    return ss.str();
}

string _assert_not_null(string object)
{
    stringstream ss;
    ss << "assert(NULL!=" << object << ")";
    return ss.str();
}

string _assign(string lvalue, string rvalue)
{
    stringstream ss;
    ss << lvalue << " = " << rvalue;
    return ss.str();
}

string _declare(string type, string variable)
{
    stringstream ss;
    ss << type << " " << variable << _end();
    return ss.str(); 
}

string _declare(string type, string variable, string expr)
{
    stringstream ss;
    ss << type << " " << variable << " = " << expr << _end();
    return ss.str(); 
}

string _cast(string type, string object)
{
    stringstream ss;
    ss << "(" << type << ")" << "(" << object << ")";
    return ss.str();
}

string _end(void)
{
    stringstream ss;
    ss << ";" << endl;
    return ss.str();
}

string _line(string object)
{
    stringstream ss;
    ss << object << _end();
    return ss.str();
}

string _add(string left, string right)
{
    stringstream ss;
    ss << left << " + " << right;
    return ss.str();
}

string _sub(string left, string right)
{
    stringstream ss;
    ss << left << " - " << right;
    return ss.str();
}

string _mul(string left, string right)
{
    stringstream ss;
    ss << left << " * " << right;
    return ss.str();
}

string _div(string left, string right)
{
    stringstream ss;
    ss << left << " / " << right;
    return ss.str();
}

string _mod(string left, string right)
{
    stringstream ss;
    ss << left << " % " << right;
    return ss.str();
}

string _inc(string object)
{
    stringstream ss;
    ss << "++("<< object << ")";
    return ss.str();
}

string _inc_post(string object)
{
    stringstream ss;
    ss << "("<< object << ")++";
    return ss.str();
}

}}}}
