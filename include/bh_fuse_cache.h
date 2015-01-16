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

#ifndef __BH_FUSE_CACHE_H
#define __BH_FUSE_CACHE_H

#include <bh.h>
#include <string>
#include <map>
#include <boost/foreach.hpp>
#include <boost/unordered_map.hpp>

namespace bohrium {

    //Forward declarations
    struct BatchHash;

    /* A class that represets a hash of a single instruction */
    struct InstrHash: public std::string
    {
        InstrHash(BatchHash &batch, const bh_instruction &instr);
    };

    /* A class that represets a hash of a instruction batch
     * (aka instruction list) */
    struct BatchHash
    {
        uint64_t base_id_count;
        std::map<const bh_base*, uint64_t> base2id;
        uint64_t hash_key;

        /* Construct a BatchHash instant based on the instruction list */
        BatchHash(const std::vector<bh_instruction> &instr_list);

        /* Returns the hash value */
        uint64_t hash() const;
    };

    /* A class that represets a cache of calculated 'instr_indexes' */
    class FuseCache
    {
        typedef typename std::vector<std::vector<uint64_t> > InstrIndexesList;
        typedef typename boost::unordered_map<uint64_t, InstrIndexesList> CacheMap;

        //The map from BatchHash to a list of 'instr_indexes'
        CacheMap cache;

    public:

        /* Insert a 'kernel_list' into the fuse cache
         *
         * @hash  The hash of the batch (aka instruction list)
         *        that matches the 'kernel_list'
         */
        void insert(const BatchHash &hash,
                    const std::vector<bh_ir_kernel> &kernel_list);

        /* Lookup a 'kernel_list' in the cache
         *
         * @hash   The hash of the batch (aka instruction list)
         *         that matches the 'kernel_list'
         * @bhir   The BhIR associated with the batch
         * @return Whether the lookup was a success or not
         */
        bool lookup(const BatchHash &hash,
                    bh_ir &bhir,
                    std::vector<bh_ir_kernel> &kernel_list);
    };

} //namespace bohrium

#endif

