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
#include <bh.h>
#include <vector>
#include <iostream>
#include "bh_ir.h"

using namespace std;
namespace io = boost::iostreams;

/* Creates a Bohrium Internal Representation (BhIR) from a instruction list.
*
* @ninstr      Number of instructions
* @instr_list  The instruction list
*/
bh_ir::bh_ir(bh_intp ninstr, const bh_instruction instr_list[])
{
    bh_ir::instr_list = vector<bh_instruction>(instr_list, &instr_list[ninstr]);
}

/* Creates a BhIR from a serialized BhIR.
*
* @bhir The BhIr serialized as a char array or vector
*/
bh_ir::bh_ir(const char bhir[], bh_intp size)
{
    io::basic_array_source<char> source(bhir,size);
    io::stream<io::basic_array_source <char> > input_stream(source);
    boost::archive::binary_iarchive ia(input_stream);
    ia >> *this;
}

/* Serialize the BhIR object into a char buffer
*  (use the bh_ir constructor for deserialization)
*
*  @buffer   The char vector to serialize into
*/
void bh_ir::serialize(vector<char> &buffer)
{
    io::stream<io::back_insert_device<vector<char> > > output_stream(buffer);
    boost::archive::binary_oarchive oa(output_stream);
    oa << *this;
    output_stream.flush();
}
