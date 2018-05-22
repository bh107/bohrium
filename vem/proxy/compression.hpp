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

#pragma once

#include <bh_view.hpp>

namespace bohrium {
class Compression {
    struct Stat {
        uint64_t total_raw;
        uint64_t total_compressed;

        Stat(uint64_t total_raw, uint64_t total_compressed) : total_raw(total_raw),
                                                              total_compressed(total_compressed) {}
    };

    std::map<std::string, std::vector<Stat> > stat_per_codex;

public:
    Compression() = default;

    /** Compress `ary`
     *
     * @param ary    The array view to compress, the view MUST represent the whole base array and be contiguous
     * @param param  A string of parameters to parsed through to the compress library
     * @return       The compressed bytes
     */
    std::vector<unsigned char> compress(const bh_view &ary, const std::string &param);

    /** Compress `ary`
     *
     * @param ary    The array base to compress
     * @param param  A string of parameters to parsed through to the compress library
     * @return       The compressed bytes
     */
    std::vector<unsigned char> compress(const bh_base &ary, const std::string &param);

    /** Uncompress `data` into `ary`
     *
     * @param data   The byte of compressed data
     * @param ary    The output array
     * @param param  A string of parameters to parsed through to the compress library
     */
    void uncompress(const std::vector<unsigned char> &data, bh_view &ary, const std::string &param);

    /** Uncompress `data` into `ary`
     *
     * @param data   The byte of compressed data
     * @param ary    The output array
     * @param param  A string of parameters to parsed through to the compress library
     */
    void uncompress(const std::vector<unsigned char> &data, bh_base &ary, const std::string &param);

    /** Pretty print statistics
     *
     * @return The printed string
     */
    std::string pprintStats() const;

    /** Pretty print detailed statistics, which include statistics for each package transfer
     *
     * @return The printed string
     */
    std::string pprintStatsDetail() const;
};
}