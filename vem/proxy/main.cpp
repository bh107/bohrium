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
#include <bh_serialize.hpp>

#include "comm.hpp"

using namespace bohrium;
using namespace component;
using namespace std;

namespace {
class Impl : public ComponentImpl {
private:
    CommFrontend comm_front;
public:
    Impl(int stack_level) : ComponentImpl(stack_level),
                            comm_front(stack_level,
                                       config.defaultGet<string>("address", "127.0.0.1"),
                                       config.defaultGet<int>("port", 4200)) {


    }

    ~Impl() { }

    void execute(bh_ir *bhir) { comm_front.execute(*bhir); }

    void extmethod(const std::string &name, bh_opcode opcode) {
        throw runtime_error("[PROXY-VEM] extmethod() not implemented!");
    };
};
} //Unnamed namespace


extern "C" ComponentImpl* create(int stack_level) {
    return new Impl(stack_level);
}
extern "C" void destroy(ComponentImpl* self) {
    delete self;
}

