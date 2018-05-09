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
#include <fstream>
#include <sstream>

#include <bh_component.hpp>

using namespace bohrium;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImpl {
  private:
    int count = 0;
  public:
    Impl(int stack_level) : ComponentImpl(stack_level) {};
    ~Impl() override = default;
    void execute(BhIR *bhir) override {
        stringstream ss;
        ss << "trace-" << count << ".txt";
        cout << "pprint-filter: writing trace('" << ss.str() << "')." << endl;
        ofstream f(ss.str());
        f << "Trace " << count++ << " (syncs:";
        for (const bh_base *b: bhir->getSyncs()) {
            f << " a" << b->get_label();
        }
        f << "):" << endl;
        for (const bh_instruction &instr: bhir->instr_list) {
            f << instr << endl;
         // f << instr.pprint(false) << endl;
        }
        f << endl;
        f.close();
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
