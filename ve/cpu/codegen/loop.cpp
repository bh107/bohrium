#include <sstream>
#include <string>
#include <map>
#include "codegen.hpp"

using namespace std;
using namespace kp::core;

namespace kp{
namespace engine{
namespace codegen{

Loop::Loop(Plaid& plaid, std::string template_fn) : Codeblock(plaid, template_fn) {}

void Loop::init(string source)
{
    init_ << source;    
}

void Loop::cond(string source)
{
    cond_ << source;    
}

void Loop::incr(string source)
{
    incr_ << source;    
}

std::string Loop::init(void)
{
    return init_.str();
}

std::string Loop::cond(void)
{
    return cond_.str();
}

std::string Loop::incr(void)
{
    return incr_.str();
}

std::string Loop::emit(void)
{
    std::map<string, string> subjects;

    subjects["PROLOG"] = prolog();
    subjects["EPILOG"] = epilog();
    subjects["PRAGMA"] = pragma();

    subjects["INIT"] = init();
    subjects["COND"] = cond();
    subjects["INCR"] = incr();

    subjects["HEAD"] = head();
    subjects["BODY"] = body();
    subjects["FOOT"] = foot();

    return plaid_.fill(template_fn_, subjects);
}

}}}
