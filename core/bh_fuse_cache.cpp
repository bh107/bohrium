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

#include <string>
#include <cstring>
#include <bh.h>
#include <boost/unordered_map.hpp>
#include "bh_fuse_cache.h"

using namespace std;
using namespace boost;

namespace bohrium {

    //Constructor of the InstrHash class
    InstrHash::InstrHash(BatchHash &batch, const bh_instruction &instr)
    {
        BOOST_FOREACH(const bh_view &view, instr.operand)
        {
            if(bh_is_constant(&view))//We ignore constants
                continue;

            //Hash the base array pointer
            uint64_t base_id;
            map<const bh_base*, uint64_t>::iterator it = batch.base2id.find(view.base);
            if(it != batch.base2id.end())
            {
                base_id = it->second;
            }
            else
            {
                base_id = batch.base_id_count++;
                batch.base2id.insert(make_pair(view.base, base_id));
            }
            this->append((char*)&base_id, sizeof(base_id));

            //Hash ndim and start
            this->append((char*)&view.ndim, sizeof(view.ndim));
            this->append((char*)&view.start, sizeof(view.start));

            //Hash shape and stride
            this->append((char*)view.shape, sizeof(bh_index)*view.ndim);
            this->append((char*)view.stride, sizeof(bh_index)*view.ndim);
        }
    }

    //Constructor of the BatchHash class
    BatchHash::BatchHash(const vector<bh_instruction> &instr_list):base_id_count(0)
    {
        BOOST_FOREACH(const bh_instruction &instr, instr_list)
        {
            this->append(InstrHash(*this, instr));
        }
    }

    static unordered_map<BatchHash, vector<vector<uint64_t> > > cache;
    void fuse_cache_insert(const BatchHash &batch,
                           const vector<bh_ir_kernel> &kernel_list)
    {
        vector<vector<uint64_t> > instr_indexes_list;
        BOOST_FOREACH(const bh_ir_kernel &kernel, kernel_list)
        {
            instr_indexes_list.push_back(kernel.instr_indexes);
        }
        cache[batch] = instr_indexes_list;
    }

    bool fuse_cache_lookup(const BatchHash &batch,
                           bh_ir &bhir,
                           vector<bh_ir_kernel> &kernel_list)
    {
        assert(kernel_list.size() == 0);
        auto it = cache.find(batch);
        if(it == cache.end())
            return false;

        BOOST_FOREACH(const vector<uint64_t> &instr_indexes, it->second)
        {
            bh_ir_kernel kernel(bhir);
            BOOST_FOREACH(uint64_t instr_idx, instr_indexes)
            {
                kernel.add_instr(instr_idx);
            }
            kernel_list.push_back(kernel);
        }
        return true;
    }
} //namespace bohrium
