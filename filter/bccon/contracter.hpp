#ifndef __BH_FILTER_COMPOSITE_CONTRACTER
#define __BH_FILTER_COMPOSITE_CONTRACTER

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

    void contract_reduction(BhIR& bhir);
    void contract_stupidmath(BhIR& bhir);
    void contract_collect(BhIR& bhir);
    void contract_muladd(BhIR& bhir);
private:
    bool repeats_;
    bool reduction_;
    bool stupidmath_;
    bool collect_;
    bool muladd_;
};

}}}
#endif
