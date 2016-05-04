#ifndef __BH_FILTER_COMPOSITE_CONTRACTER
#define __BH_FILTER_COMPOSITE_CONTRACTER

#include <bh_component.h>

namespace bohrium {
namespace filter {
namespace composite {

class Contracter
{
public:
    Contracter(bool repeats, bool reduction, bool stupidmath, bool collect);

    ~Contracter(void);

    void contract(bh_ir& bhir);

    void contract_repeats(bh_ir& bhir);
    void contract_reduction(bh_ir& bhir);
    void contract_stupidmath(bh_ir& bhir);
    void contract_collect(bh_ir& bhir);

private:
    bool repeats_;
    bool reduction_;
    bool stupidmath_;
    bool collect_;
};

}}}
#endif
