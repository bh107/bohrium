/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#include <bh_component.hpp>
#include "expander.hpp"

using namespace bohrium;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImplWithChild {
private:
    filter::bcexp::Expander expander;
public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level),
                            expander(config.defaultGet<bool>("verbose", false),
                                     config.defaultGet<int>("gc_threshold", 400),
                                     config.defaultGet<bool>("sign", true),
                                     config.defaultGet<bool>("powk", true),
                                     config.defaultGet<int>("reduce1d", 32000)) {};

    ~Impl() {}; // NB: a destructor implementation must exist
    void execute(BhIR *bhir) {
        expander.expand(*bhir);
        child.execute(bhir);
    };
};
} //Unnamed namespace

extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}
