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

    void contract(bh_ir& bhir);

    void contract_repeats(bh_ir& bhir);
    void contract_reduction(bh_ir& bhir);
    void contract_stupidmath(bh_ir& bhir);
    void contract_collect(bh_ir& bhir);
    void contract_muladd(bh_ir& bhir);
private:
    bool repeats_;
    bool reduction_;
    bool stupidmath_;
    bool collect_;
    bool muladd_;
};

}}}
#endif
