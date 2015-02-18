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

string _add_assign(string lvalue, string rvalue)
{
    stringstream ss;
    ss << lvalue << " += " << rvalue;
    return ss.str();
}

string _sub_assign(string lvalue, string rvalue)
{
    stringstream ss;
    ss << lvalue << " -= " << rvalue;
    return ss.str();
}

string _declare(string type, string variable)
{
    stringstream ss;
    ss << type << " " << variable;
    return ss.str(); 
}

string _declare_init(string type, string variable, string expr)
{
    stringstream ss;
    ss << type << " " << variable << " = " << expr;
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

string _end(string comment)
{
    stringstream ss;
    ss << "; // " << comment << endl;
    return ss.str();
}

string _line(string object)
{
    stringstream ss;
    ss << object << _end();
    return ss.str();
}

string _dec(string object)
{
    stringstream ss;
    ss << "--("<< object << ")";
    return ss.str();
}

string _dec_post(string object)
{
    stringstream ss;
    ss << "("<< object << ")--";
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
    ss << left << " - floor(" << left << " / " << right << ") * " << right;
    return ss.str();
}

string _pow(string left, string right)
{
    stringstream ss;
    ss << "pow(" << left << ", " << right << ")";
    return ss.str();
}

string _cpow(string left, string right)
{
    stringstream ss;
    ss << "cpow(" << left << ", " << right << ")";
    return ss.str();
}

string _cpowf(string left, string right)
{
    stringstream ss;
    ss << "cpowf(" << left << ", " << right << ")";
    return ss.str();
}

string _abs(string left, string right)
{
    stringstream ss;
    ss << left << " < 0.0 ? -" << right << ": " << right;
    return ss.str();
}

string _gt(string left, string right)
{
    stringstream ss;
    ss << left << " > " << right;
    return ss.str();
}

string _gteq(string left, string right)
{
    stringstream ss;
    ss << left << " >= " << right;
    return ss.str();
}

string _lt(string left, string right)
{
    stringstream ss;
    ss << left << " < " << right;
    return ss.str();
}

string _lteq(string left, string right)
{
    stringstream ss;
    ss << left << " <= " << right;
    return ss.str();
}

string _eq(string left, string right)
{
    stringstream ss;
    ss << left << " == " << right;
    return ss.str();
}

string _neq(string left, string right)
{
    stringstream ss;
    ss << left << " != " << right;
    return ss.str();
}

string _logic_and(string left, string right)
{
    stringstream ss;
    ss << left << " && " << right;
    return ss.str();
}

string _logic_or(string left, string right)
{
    stringstream ss;
    ss << left << " || " << right;
    return ss.str();
}

string _logic_xor(string left, string right)
{
    stringstream ss;
    ss << "(!" << left << " != !" << right << ")";
    return ss.str();
}

string _logic_not(string right)
{
    stringstream ss;
    ss << left << " !" << right;
    return ss.str();
}

string _bitw_and(string left, string right)
{
    stringstream ss;
    ss << left << " & " << right;
    return ss.str();
}

string _bitw_or(string left, string right)
{
    stringstream ss;
    ss << left << " | " << right;
    return ss.str();
}

string _bitw_xor(string left, string right)
{
    stringstream ss;
    ss << left << " ^ " << right;
    return ss.str();
}

string _invert(string right)
{
    stringstream ss;
    ss << "~" << right;
    return ss.str();
}

string _invertb(string right)
{
    stringstream ss;
    ss << "!" << right;
    return ss.str();
}

string _bitw_leftshift(string left, string right)
{
    stringstream ss;
    ss << "((" << left << ") << (" << right << "))";
    return ss.str();
}

string _bitw_rightshift(string left, string right)
{
    stringstream ss;
    ss << "((" << left << ") << (" << right << "))";
    return ss.str();
}

string _max(string left, string right)
{
    stringstream ss;
    ss << left << " < " << right << " ? " << right << " : " << left;
    return ss.str();
}

string _min(string left, string right)
{
    stringstream ss;
    ss << left << " < " << right << " ? " << left << " : " << right;
    return ss.str();
}

string _sin(string right)
{
    stringstream ss;
    ss << "sin(" << right << ")";
    return ss.str();
}

string _csinf(string right)
{
    stringstream ss;
    ss << "csinf(" << right << ")";
    return ss.str();
}

string _csin(string right)
{
    stringstream ss;
    ss << "csin(" << right << ")";
    return ss.str();
}

string _asinh(string right)
{
    stringstream ss;
    ss << "asinh(" << right << ")";
    return ss.str();
}

string _casinhf(string right)
{
    stringstream ss;
    ss << "casinhf(" << right << ")";
    return ss.str();
}

string _casinh(string right)
{
    stringstream ss;
    ss << "casinh(" << right << ")";
    return ss.str();
}

string _sinh(string right)
{
    stringstream ss;
    ss << "sinh(" << right << ")";
    return ss.str();
}

string _csinhf(string right)
{
    stringstream ss;
    ss << "csinhf(" << right << ")";
    return ss.str();
}

string _csinh(string right)
{
    stringstream ss;
    ss << "csinh(" << right << ")";
    return ss.str();
}

string _asin(string right)
{
    stringstream ss;
    ss << "asin(" << right << ")";
    return ss.str();
}

string _casinf(string right)
{
    stringstream ss;
    ss << "casinf(" << right << ")";
    return ss.str();
}

string _casin(string right)
{
    stringstream ss;
    ss << "casin(" << right << ")";
    return ss.str();
}

string _cos(string right)
{
    stringstream ss;
    ss << "cos(" << right << ")";
    return ss.str();
}

string _ccosf(string right)
{
    stringstream ss;
    ss << "ccosf(" << right << ")";
    return ss.str();
}

string _ccos(string right)
{
    stringstream ss;
    ss << "ccos(" << right << ")";
    return ss.str();
}

string _acos(string right)
{
    stringstream ss;
    ss << "acos(" << right << ")";
    return ss.str();
}

string _cacos(string right)
{
    stringstream ss;
    ss << "cacos(" << right << ")";
    return ss.str();
}

string _cacosf(string right)
{
    stringstream ss;
    ss << "cacosf(" << right << ")";
    return ss.str();
}

string _cosh(string right)
{
    stringstream ss;
    ss << "cosh(" << right << ")";
    return ss.str();
}

string _ccoshf(string right)
{
    stringstream ss;
    ss << "ccoshf(" << right << ")";
    return ss.str();
}

string _ccosh(string right)
{
    stringstream ss;
    ss << "ccosh(" << right << ")";
    return ss.str();
}

string _acosh(string right)
{
    stringstream ss;
    ss << "acosh(" << right << ")";
    return ss.str();
}

string _cacoshf(string right)
{
    stringstream ss;
    ss << "cacoshf(" << right << ")";
    return ss.str();
}

string _cacosh(string right)
{
    stringstream ss;
    ss << "cacosh(" << right << ")";
    return ss.str();
}

string _tan(string right)
{
    stringstream ss;
    ss << "tan(" << right << ")";
    return ss.str();
}

string _ctanf(string right)
{
    stringstream ss;
    ss << "ctanf(" << right << ")";
    return ss.str();
}

string _ctan(string right)
{
    stringstream ss;
    ss << "ctan(" << right << ")";
    return ss.str();
}

string _tanh(string right)
{
    stringstream ss;
    ss << "tanh(" << right << ")";
    return ss.str();
}

string _ctanhf(string right)
{
    stringstream ss;
    ss << "ctanhf(" << right << ")";
    return ss.str();
}

string _ctanh(string right)
{
    stringstream ss;
    ss << "ctanh(" << right << ")";
    return ss.str();
}

string _atan(string right)
{
    stringstream ss;
    ss << "atan(" << right << ")";
    return ss.str();
}

string _catanf(string right)
{
    stringstream ss;
    ss << "catanf(" << right << ")";
    return ss.str();
}

string _catan(string right)
{
    stringstream ss;
    ss << "catan(" << right << ")";
    return ss.str();
}

string _atanh(string right)
{
    stringstream ss;
    ss << "atanh(" << right << ")";
    return ss.str();
}

string _catanhf(string right)
{
    stringstream ss;
    ss << "catanhf(" << right << ")";
    return ss.str();
}

string _catanh(string right)
{
    stringstream ss;
    ss << "catanh(" << right << ")";
    return ss.str();
}

string _atan2(string left, string right)
{
    stringstream ss;
    ss << "atan2(" << left << ", " << right << ")";
    return ss.str();
}

string _exp(string right)
{
    stringstream ss;
    ss << "exp(" << right << ")";
    return ss.str();
}

string _cexpf(string right)
{
    stringstream ss;
    ss << "cexpf(" << right << ")";
    return ss.str();
}

string _cexp(string right)
{
    stringstream ss;
    ss << "cexp(" << right << ")";
    return ss.str();
}

string _exp2(string right)
{
    stringstream ss;
    ss << "pow(2, " << right << ")";
    return ss.str();
}

string _cexp2f(string right)
{
    stringstream ss;
    ss << "cpowf(2, " << right << ")";
    return ss.str();
}

string _cexp2(string right)
{
    stringstream ss;
    ss << "cpow(2, " << right << ")";
    return ss.str();
}

string _expm1(string right)
{
    stringstream ss;
    ss << "expm1(" << right << ")";
    return ss.str();
}

string _log(string right)
{
    stringstream ss;
    ss << "log(" << right << ")";
    return ss.str();
}

string _clogf(string right)
{
    stringstream ss;
    ss << "clogf(" << right << ")";
    return ss.str();
}

string _clog(string right)
{
    stringstream ss;
    ss << "clog(" << right << ")";
    return ss.str();
}

string _log10(string right)
{
    stringstream ss;
    ss << "log(" << right << ")/log(10)";
    return ss.str();
}

string _clog10f(string right)
{
    stringstream ss;
    ss << "clog10f(" << right << ")/log(10)";
    return ss.str();
}

string _clog10(string right)
{
    stringstream ss;
    ss << "clog10(" << right << ")/log(10)";
    return ss.str();
}

string _log2(string right)
{
    stringstream ss;
    ss << "log2(" << right << ")";
    return ss.str();
}

string _log1p(string right)
{
    stringstream ss;
    ss << "log1p(" << right << ")";
    return ss.str();
}

string _sqrt(string right)
{
    stringstream ss;
    ss << "sqrt(" << right << ")";
    return ss.str();
}

string _csqrt(string right)
{
    stringstream ss;
    ss << "csqrt(" << right << ")";
    return ss.str();
}

string _csqrtf(string right)
{
    stringstream ss;
    ss << "csqrtf(" << right << ")";
    return ss.str();
}


string _ceil(string right)
{
    stringstream ss;
    ss << "ceil(" << right << ")";
    return ss.str();
}

string _trunc(string right)
{
    stringstream ss;
    ss << "trunc(" << right << ")";
    return ss.str();
}

string _floor(string right)
{
    stringstream ss;
    ss << "floor(" << right << ")";
    return ss.str();
}

string _rint(string right)
{
    stringstream ss;
    ss << "(" << right << " > 0.0) ? "
       << "floor(" << right << " + 0.5) : "
       << "ceil(" << right << " - 0.5)";
    return ss.str();
}

string _range(void)
{
    // NOTE: eidx is only defined in ewise.cont template
    stringstream ss;
    ss << "eidx";   
    return ss.str();
}

string _random(string left, string right)
{
    //
    // NOTE: eidx is only defined in ewise.cont template
    //
    // NOTE: This is a very funky implementation...
    //       the union_type (philox2x32_as1x64_t) is used as a means to 
    //       generate 64bit numbers using the philox2x32 routing.
    //
    //       Casting and anonymous unions are relied upon to avoid
    //       naming the naming intermediate variables that could cause
    //       collision.
    //
    //       Wrapping this into a nice function is avoided since we do
    //       wish to pay the overhead of calling a function for each
    //       element in the array.
    //
    stringstream ss;
    ss  << "("
        << "    (philox2x32_as_1x64_t)"
        << "    philox2x32("
        << "        ((philox2x32_as_1x64_t)( " << right << " + eidx)).orig,"
        << "        (philox2x32_key_t){ { " << left << " } }"
        << "    )"
        << ").combined;";
    return ss.str();
}

string _isnan(string right)
{
    stringstream ss;
    ss << "isnan(" << right << ")";
    return ss.str();
}

string _isinf(string right)
{
    stringstream ss;
    ss << "isinf(" << right << ")";
    return ss.str();
}

string _creal(string right)
{
    stringstream ss;
    ss << "creal(" << right << ")";
    return ss.str();
}

string _crealf(string right)
{
    stringstream ss;
    ss << "crealf(" << right << ")";
    return ss.str();
}

string _cimagf(string right)
{
    stringstream ss;
    ss << "cimagf(" << right << ")";
    return ss.str();
}

string _cimag(string right)
{
    stringstream ss;
    ss << "cimag(" << right << ")";
    return ss.str();
}

}}}}
