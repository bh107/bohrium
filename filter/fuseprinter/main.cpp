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

#include <sstream>

#include <bh_component.hpp>
#include <bh_dag.hpp>

using namespace bohrium;
using namespace bohrium::dag;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImplWithChild {
  private:
    int count = 0;
  public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level) {};
    ~Impl() {}; // NB: a destructor implementation must exist
    void execute(bh_ir *bhir) {
        GraphDW dag;
        from_kernels(bhir->kernel_list, dag);
        vector<GraphDW> dags;
        split(dag, dags);
        int i=0;
        for(GraphDW &d: dags)
        {
            stringstream ss;
            ss << "dag-" << count << "-" << i << ".dot";
            pprint(d, ss.str().c_str());
        }
        stringstream ss;
        ss << "dag-" << count++ << ".dot";
        cout << "fuseprinter: writing dag('" << ss.str() << "')." << endl;
        pprint(dag, ss.str().c_str());
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
