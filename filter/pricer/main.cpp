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

#include <iostream>

#include <bh_component.hpp>
#include <bh_dag.hpp>

using namespace bohrium;
using namespace dag;
using namespace component;
using namespace std;


namespace {
class Impl : public ComponentImplWithChild {
  private:
    int64_t sum=0, sum_uv=0, sum_wedges=0;
  public:
    Impl(int stack_level) : ComponentImplWithChild(stack_level) {};
    ~Impl() {
        cout << "[PRICER-FILTER] total cost: " << sum \
         << " (" << sum_uv << " bytes, "\
         << sum_wedges << " wedges)" << endl;
    };
    void execute(bh_ir *bhir) {
        if(bhir->kernel_list.size() > 0) {

            GraphDW dag;
            from_kernels(bhir->kernel_list, dag);
            sum += dag_cost(dag.bglD());
            sum_uv += dag_cost(dag.bglD(), bohrium::UNIQUE_VIEWS);
            sum_wedges += num_edges(dag.bglW());

            if (not dag_validate(dag)) {
                throw runtime_error("[PRICER-FILTER] Invalid BhIR! ");
            }
        }
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
