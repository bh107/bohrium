#include <sstream>
#include <string>
#include <map>
#include "codegen.hpp"

using namespace std;
using namespace kp::core;

namespace kp{
namespace engine{
namespace codegen{

Codeblock::Codeblock(Plaid& plaid, std::string template_fn) : plaid_(plaid), template_fn_(template_fn) {}

void Codeblock::prolog(string source)
{
    prolog_ << source;    
}

void Codeblock::epilog(string source)
{
    epilog_ << source;    
}

void Codeblock::pragma(string source)
{
    pragma_ << source;    
}

void Codeblock::head(string source)
{
    head_ << source;    
}

void Codeblock::body(string source)
{
    body_ << source;    
}

void Codeblock::foot(string source)
{
    foot_ << source;    
}

std::string Codeblock::prolog(void)
{
    return prolog_.str();
}

std::string Codeblock::epilog(void)
{
    return epilog_.str();
}

std::string Codeblock::pragma(void)
{
    return pragma_.str();
}

std::string Codeblock::head(void)
{
    return head_.str();
}

std::string Codeblock::body(void)
{
    return body_.str();
}

std::string Codeblock::foot(void)
{
    return foot_.str();
}

std::string Codeblock::emit(void)
{
    std::map<string, string> subjects;

    subjects["PROLOG"] = prolog();
    subjects["EPILOG"] = epilog();
    subjects["PRAGMA"] = pragma();
    subjects["HEAD"] = head();
    subjects["BODY"] = body();
    subjects["FOOT"] = foot();

    return plaid_.fill(template_fn_, subjects);
}

}}}
