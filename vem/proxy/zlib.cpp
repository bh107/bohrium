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

#include <zlib.h>
#include <vector>
#include <stdexcept>
#include <cassert>
#include "zlib.hpp"

std::vector<unsigned char> zlib_compress(void *data, uint64_t nbytes) {
    uLongf compressed_size = compressBound(nbytes);
    std::vector<unsigned char> ret(compressed_size);
    int err = compress(&ret[0], &compressed_size, (Bytef *) data, nbytes);
    if (err != Z_OK) {
        throw std::runtime_error("zlib compress(): failed");
    }
    ret.resize(compressed_size);
    return ret;
}

void zlib_uncompress(const std::vector<unsigned char> &data, void *dest, uint64_t dest_nbytes) {
    uLongf uncompressed_size = dest_nbytes;
    int err = uncompress((Bytef *) dest, &uncompressed_size, (Bytef *) (&data[0]), data.size());
    if (err != Z_OK) {
        throw std::runtime_error("zlib uncompress(): failed");
    }
    assert(dest_nbytes == uncompressed_size);
}