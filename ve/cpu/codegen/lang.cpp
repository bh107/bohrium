#include <sstream>
#include <string>
#include "codegen.hpp"

string ref(string object)
{
    stringstream ss;
    ss << "&" << object;
    return ss.str();
}

string deref(string object)
{
    stringstream ss;
    ss << "*" << object;
    return ss.str();
}

string ptr_type(string object)
{
    stringstream ss;
    ss << object << "*";
    return ss.str();
}

string const_type(string object)
{
    stringstream ss;
    ss << "const " << object;
    return ss.str();
}

string assert_not_null(string object)
{
    stringstream ss;
    ss << "assert(NULL!=" << object << ")";
    return ss.str();
}

string declare_var(text object)
{
    stringstream ss;
    
    return ss.str();
}

string declare_ptr_var(text object)
{
    stringstream ss;
    return ss.str();
}

string end(void)
{
    stringstream ss;
    ss << ";" << endl;
    return ss.str();
}

