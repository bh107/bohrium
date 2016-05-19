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

#ifndef __BH_IR_FUSER_H
#define __BH_IR_FUSER_H

#include <exception>

#include <bh_component.hpp>
#include <bh_ir.hpp>
#include <bh_fuse_cache.hpp>

namespace bohrium {

// Representation of a fuser implementation.
// This is purely for convenience since all fusers shares some common traits
class ComponentFuser : public component::ComponentImplWithChild {
private:
    // All fusers have a cache
    FuseCache _cache;
public:
    ComponentFuser(int stack_level) : ComponentImplWithChild(stack_level),
                                      _cache(config) {};
    virtual ~ComponentFuser() {};

    // On cache miss, we call this function thus anyone
    // inherent from this class should implement it
    virtual void do_fusion(bh_ir &bhir) = 0;

    // All fusers checks the cache before doing fusion
    // If cache miss, we call the do_fuction() function, which anyone
    // inherent from this class should implement
    virtual void execute(bh_ir *bhir) {

        if(bhir->kernel_list.size() != 0)
            throw std::logic_error("The kernel_list is not empty!");

        if(_cache.enabled) {
            BatchHash hash(bhir->instr_list);
            if(not _cache.lookup(hash, *bhir, bhir->kernel_list)) {//Fuse cache miss!
                do_fusion(*bhir);
                _cache.insert(hash, bhir->kernel_list);
            }
        } else {
            do_fusion(*bhir);
        }
        child.execute(bhir);
    }
};

} //namespace bohrium

#endif

