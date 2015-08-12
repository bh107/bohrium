#include <sstream>
#include "program.hpp"

using namespace std;

namespace kp{
namespace core{

const char Program::TAG[] = "Program";

Program::Program(int64_t length)
{
    program_.capacity = length;
    program_.ntacs = 0;
    program_.tacs = new kp_tac[program_.capacity];
}

Program::~Program(void)
{
    delete[] program_.tacs;
}

string Program::text_meta(void)
{
    stringstream ss;
    ss << "[";
    ss << "capacity=" << capacity() << ",";
    ss << "size=" << size();
    ss << "]";
    ss << endl;

    return ss.str();
}

void Program::clear(void)
{
    program_.ntacs = 0;
}

size_t Program::capacity(void)
{
    return program_.capacity;
}

size_t Program::size(void)
{
    return program_.ntacs;
}

kp_tac& Program::operator[](size_t tac_idx)
{
    return program_.tacs[tac_idx];
}

kp_tac* Program::tacs(void)
{
    return program_.tacs;
}

kp_program& Program::meta(void)
{
    return program_;
}

}}

