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

#include "serialize.hpp"

#include <set>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>
#include <bh_util.hpp>

using namespace std;
using namespace boost;

namespace msg {

Header::Header(const std::vector<char> &buffer)//Deserialize constructor
{
    assert(buffer.size() >= HeaderSize);

    //Interpret the buffer as a Type and a body size
    const Type *type = reinterpret_cast<const Type *>(&buffer[0]);
    const size_t *body_size = reinterpret_cast<const size_t *>(type + 1);

    //Write from buffer
    this->type = *type;
    this->body_size = *body_size;
}

void Header::serialize(std::vector<char> &buffer) {
    //Make room for the Header data
    buffer.resize(buffer.size() + HeaderSize);

    //Interpret the buffer as a Type and a body size
    Type *type = reinterpret_cast<Type *>(&buffer[0]);
    size_t *body_size = reinterpret_cast<size_t *>(type + 1);

    //Write to buffer
    *type = this->type;
    *body_size = this->body_size;
}

Init::Init(const std::vector<char> &buffer) {
    // Wrap 'buffer' in an input stream
    iostreams::basic_array_source<char> source(&buffer[0], buffer.size());
    iostreams::stream<iostreams::basic_array_source<char> > input_stream(source);
    archive::binary_iarchive ia(input_stream);

    // Deserialize the component name
    ia >> this->stack_level;
}

void Init::serialize(std::vector<char> &buffer) {
    // Wrap 'buffer' in an output stream
    iostreams::stream<iostreams::back_insert_device<vector<char> > > output_stream(buffer);
    archive::binary_oarchive oa(output_stream);

    //Serialize the component name
    oa << this->stack_level;
}

GetData::GetData(const std::vector<char> &buffer) {
    // Wrap 'buffer' in an input stream
    iostreams::basic_array_source<char> source(&buffer[0], buffer.size());
    iostreams::stream<iostreams::basic_array_source<char> > input_stream(source);
    archive::binary_iarchive ia(input_stream);

    size_t b;
    ia >> b;
    this->base = reinterpret_cast<bh_base *>(b);
    ia >> this->nullify;
}

void GetData::serialize(std::vector<char> &buffer) {
    // Wrap 'buffer' in an output stream
    iostreams::stream<iostreams::back_insert_device<vector<char> > > output_stream(buffer);
    archive::binary_oarchive oa(output_stream);

    size_t b = reinterpret_cast<size_t>(this->base);
    oa << b;
    oa << this->nullify;
}

MemCopy::MemCopy(const std::vector<char> &buffer) {
    // Wrap 'buffer' in an input stream
    iostreams::basic_array_source<char> source(&buffer[0], buffer.size());
    iostreams::stream<iostreams::basic_array_source<char> > input_stream(source);
    archive::binary_iarchive ia(input_stream);

    ia >> this->src;
    size_t b;
    ia >> b;
    this->src.base = reinterpret_cast<bh_base *>(b);
    ia >> this->param;
}

void MemCopy::serialize(std::vector<char> &buffer) {
    // Wrap 'buffer' in an output stream
    iostreams::stream<iostreams::back_insert_device<vector<char> > > output_stream(buffer);
    archive::binary_oarchive oa(output_stream);

    oa << this->src;
    size_t b = reinterpret_cast<size_t>(this->src.base);
    oa << b;
    oa << this->param;
}

Message::Message(const std::vector<char> &buffer) {
    // Wrap 'buffer' in an input stream
    iostreams::basic_array_source<char> source(&buffer[0], buffer.size());
    iostreams::stream<iostreams::basic_array_source<char> > input_stream(source);
    archive::binary_iarchive ia(input_stream);

    ia >> msg;
}

void Message::serialize(std::vector<char> &buffer) {
    // Wrap 'buffer' in an output stream
    iostreams::stream<iostreams::back_insert_device<vector<char> > > output_stream(buffer);
    archive::binary_oarchive oa(output_stream);

    oa << msg;
}

}