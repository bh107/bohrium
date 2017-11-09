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

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/set.hpp>

#include <bh_ir.hpp>
#include <bh_util.hpp>

using namespace std;
using namespace boost;
namespace io = boost::iostreams;


BhIR::BhIR(const std::vector<char> &serialized_archive, std::map<const bh_base*, bh_base> &remote2local,
           vector<bh_base*> &data_recv, set<bh_base*> &frees) {

    // Wrap 'serialized_archive' in an input stream
    iostreams::basic_array_source<char> source(&serialized_archive[0], serialized_archive.size());
    iostreams::stream<iostreams::basic_array_source<char> > input_stream(source);
    archive::binary_iarchive ia(input_stream);

    // Load number of repeats and the repeat condition
    ia >> _nrepeats;
    {
        size_t t;
        ia >> t;
        _repeat_condition = reinterpret_cast<bh_base*>(t);
    }

    // Load the instruction list
    ia >> instr_list;

    // Load the set of syncs
    {
        vector<size_t> base_as_int;
        ia >> base_as_int;
        for(size_t base: base_as_int) {
            _syncs.insert(reinterpret_cast<bh_base*>(base));
        }
    }

    // Load the new base arrays
    std::vector<bh_base> news;
    ia >> news;

    // Find all freed base arrays (remote base pointers)
    for (const bh_instruction &instr: instr_list) {
        if (instr.opcode == BH_FREE) {
            frees.insert(instr.operand[0].base);
        }
    }

    // Add the new base array to 'remote2local' and to 'data_recv'
    size_t new_base_count = 0;
    for (const bh_instruction &instr: instr_list) {
        for (const bh_view &v: instr.operand) {
            if (bh_is_constant(&v))
                continue;
            if (not util::exist(remote2local, v.base)) {
                assert(new_base_count < news.size());
                remote2local[v.base] = news[new_base_count++];
                if (remote2local[v.base].data != nullptr)
                    data_recv.push_back(&remote2local[v.base]);
            }
        }
    }
    assert(new_base_count == news.size());

    // Update all base pointers to point to the local bases
    for (bh_instruction &instr: instr_list) {
        for (bh_view &v: instr.operand) {
            if (not bh_is_constant(&v)) {
                v.base = &remote2local.at(v.base);
            }
        }
    }
    // Update all base pointers in the bhir's `_syncs` set
    {
        set<bh_base*> syncs_as_local_ptr;
        for (bh_base *base: _syncs) {
            if (util::exist(remote2local, base)) {
                syncs_as_local_ptr.insert(&remote2local.at(base));
            }
        }
        _syncs = std::move(syncs_as_local_ptr);
    }
    // Update the `_repeat_condition` pointer
    if (_repeat_condition != nullptr) {
        _repeat_condition = &remote2local.at(_repeat_condition);
    }

}

std::vector<char> BhIR::write_serialized_archive(set<bh_base *> &known_base_arrays, vector<bh_base *> &new_data) {

    // Find new base arrays in 'bhir', which the de-serializing component should know about, and their data (if any)
    vector<bh_base> new_bases; // New base arrays in the order they appear in the instruction list
    for (bh_instruction &instr: instr_list) {
        for (const bh_view &v: instr.operand) {
            if (not bh_is_constant(&v) and not util::exist(known_base_arrays, v.base)) {
                new_bases.push_back(*v.base);
                known_base_arrays.insert(v.base);
                if (v.base->data != nullptr) {
                    new_data.push_back(v.base);
                }
            }
        }
    }

    // Wrap 'ret' in an output stream
    std::vector<char> ret;
    iostreams::stream<iostreams::back_insert_device<vector<char> > > output_stream(ret);
    archive::binary_oarchive oa(output_stream);

    // Write number of repeats and the repeat condition
    oa << _nrepeats;
    if (_repeat_condition != nullptr and util::exist(known_base_arrays, _repeat_condition)) {
        size_t t = reinterpret_cast<size_t >(_repeat_condition);
        oa << t;
    } else {
        size_t t = 0;
        oa << t;
    }

    // Write the instruction list
    oa << instr_list;

    vector<size_t> base_as_int;
    for(bh_base *base: _syncs) {
        base_as_int.push_back(reinterpret_cast<size_t>(base));
    }
    oa << base_as_int;
    oa << new_bases;
    return ret;
}
