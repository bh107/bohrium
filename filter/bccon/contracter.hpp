#pragma once

#include <bh_component.hpp>

namespace bohrium {
namespace filter {
namespace bccon {

extern bool __verbose;
extern void verbose_print(std::string str);

class Contracter
{
public:
    Contracter(bool verbose, bool repeats, bool reduction, bool stupidmath, bool collect, bool muladd);

    ~Contracter(void);

    void contract(BhIR& bhir);

    void reduction(BhIR& bhir);
    void stupidmath(BhIR& bhir);
    void collect(BhIR& bhir);
    void muladd(BhIR& bhir);
private:
    bool repeats_;
    bool reduction_;
    bool stupidmath_;
    bool collect_;
    bool muladd_;
};

}}}
