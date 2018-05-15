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

#include <bh_base.hpp>
#include <bh_main_memory.hpp>
#include <zlib.h>
#include "compression.hpp"


using namespace std;

std::vector<unsigned char> compress(const bh_view &ary) {
    std::vector<unsigned char> ret;
    if (not bh_is_contiguous(&ary) or bh_nelements(ary) != ary.base->nelem) {
        throw std::runtime_error("compress(): `ary` must be contiguous and represent the whole of its base");
    }
    if (ary.base->data == nullptr) {
        throw std::runtime_error("compress(): `ary` data is NULL");
    }
    uLongf compress_size = compressBound(ary.base->nbytes());
    ret.resize(compress_size);
    int err = compress(&ret[0], &compress_size, (Bytef *) ary.base->data, ary.base->nbytes());
    if (err != Z_OK) {
        throw std::runtime_error("zlib compress(): failed");
    }
    ret.resize(compress_size);
    return ret;
}

std::vector<unsigned char> compress(const bh_base &ary) {
    auto &a = const_cast<bh_base&>(ary);
    const bh_view view{a}; // View of the whole base
    return compress(view);
}

void uncompress(const std::vector<unsigned char> &data, bh_view &ary) {
    if (not bh_is_contiguous(&ary) or bh_nelements(ary) != ary.base->nelem) {
        throw std::runtime_error("uncompress(): `ary` must be contiguous and represent the whole of its base");
    }
    if (data.empty()) {
        throw std::runtime_error("uncompress(): `data` is empty!");
    }
    bh_data_malloc(ary.base);
    size_t size = ary.base->nbytes();
    int err = uncompress((Bytef *) ary.base->data, &size, (Bytef *) (&data[0]), data.size());
    if (err != Z_OK) {
        throw std::runtime_error("zlib uncompress(): failed");
    }
    assert(size == ary.base->nbytes());
}

void uncompress(const std::vector<unsigned char> &data, bh_base &ary) {
    bh_view view{ary}; // View of the whole base
    uncompress(data, view);
}

