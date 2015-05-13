#ifndef __BH_FILTER_COMPOSITE_EXPANDER
#define __BH_FILTER_COMPOSITE_EXPANDER
#include "bh.h"

namespace bohrium {
namespace filter {
namespace composite {

class Expander
{
public:
    Expander();
    void expand(bh_ir& bhir);

private:

    static const char TAG[];
};

}}}
#endif
