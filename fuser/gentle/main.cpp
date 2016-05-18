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
#include <bh_fuser.hpp>
#include <bh_dag.hpp>

using namespace bohrium;
using namespace component;
using namespace dag;
using namespace std;

namespace {
class Impl : public ComponentFuser {
public:
    Impl(int stack_level) : ComponentFuser(stack_level) {}
    ~Impl() {};
    void do_fusion(bh_ir &bhir) {

        GraphDW dag;
        from_bhir(bhir, dag);
        vector<GraphDW> dags;
        split(dag, dags);
        assert(dag_validate(bhir, dags));
        for(GraphDW &d: dags) {
            fuse_gently(d);
        }
        assert(dag_validate(bhir, dags));
        for(GraphDW &d: dags) {
            fill_kernel_list(d.bglD(), bhir.kernel_list);
        }
    }
};
} //Unnamed namespace

extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}