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

#include <bh_serialize.hpp>

#include <set>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/stream_buffer.hpp>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/device/back_inserter.hpp>

using namespace std;
using namespace boost;
namespace bohrium {
namespace serialize {

Header::Header(const std::vector<char> &buffer)//Deserialize constructor
{
    assert(buffer.size() >= HeaderSize);

    //Interpret the buffer as a Type and a body size
    const Type *type = reinterpret_cast<const Type*>(&buffer[0]);
    const size_t *body_size = reinterpret_cast<const size_t*>(type+1);

    //Write from buffer
    this->type = *type;
    this->body_size = *body_size;
}

void Header::serialize(std::vector<char> &buffer)
{
    //Make room for the Header data
    buffer.resize(buffer.size()+HeaderSize);

    //Interpret the buffer as a Type and a body size
    Type *type = reinterpret_cast<Type*>(&buffer[0]);
    size_t *body_size = reinterpret_cast<size_t*>(type+1);

    //Write to buffer
    *type = this->type;
    *body_size = this->body_size;
}

Init::Init(const std::vector<char> &buffer)//Deserialize constructor
{
    //Wrap 'buffer' in an io stream
    iostreams::basic_array_source<char> source(&buffer[0], buffer.size());
    iostreams::stream<iostreams::basic_array_source <char> > input_stream(source);
    archive::binary_iarchive ia(input_stream);

    //Deserialize the component name
    ia >> this->component_name;
}

void Init::serialize(std::vector<char> &buffer)
{
    //Wrap 'buffer' in an io stream
    iostreams::stream<iostreams::back_insert_device<vector<char> > > output_stream(buffer);
    archive::binary_oarchive oa(output_stream);

    //Serialize the component name
    oa << this->component_name;
}

void ExecuteFrontend::serialize(const bh_ir &bhir, vector<char> &buffer, vector<bh_base*> &data_send, vector<bh_base*> &data_recv)
{
    //Wrap 'buffer' in an io stream
    iostreams::stream<iostreams::back_insert_device<vector<char> > > output_stream(buffer);
    archive::binary_oarchive oa(output_stream);

    //Serialize the BhIR
    oa << bhir;

    //Serialize the new base arrays in 'bhir' and find base arrays that have data we must send
    vector<bh_base> new_bases;//New base arrays in the order they appear in the instruction list
    for(const bh_instruction &instr: bhir.instr_list)
    {
        const int nop = bh_noperands(instr.opcode);
        for(int i=0; i<nop; ++i)
        {
            const bh_view &v = instr.operand[i];
            if(bh_is_constant(&v))
                continue;
            if(known_base_arrays.find(v.base) == known_base_arrays.end())
            {
                new_bases.push_back(*v.base);
                known_base_arrays.insert(v.base);
                if(v.base->data != NULL)
                    data_send.push_back(v.base);
            }
        }
    }

    //Serialize the new base arrays in the 'bhir'
    oa << new_bases;

    //Update 'known_base_arrays' and 'data_recv'
    for(const bh_instruction &instr: bhir.instr_list)
    {
        assert(instr.opcode >= 0);
        switch(instr.opcode)
        {
            case BH_DISCARD:
            {
                known_base_arrays.erase(instr.operand[0].base);
                break;
            }
            case BH_SYNC:
            {
                data_recv.push_back(instr.operand[0].base);
                break;
            }
            default:{}
        }
    }
}

void ExecuteFrontend::cleanup(bh_ir &bhir)
{
    for(const bh_instruction &instr: bhir.instr_list)
    {
        assert(instr.opcode >= 0);
        switch(instr.opcode)
        {
            case BH_FREE:
            {
                bh_data_free(instr.operand[0].base);
                break;
            }
            default:{}
        }
    }
}

bh_ir ExecuteBackend::deserialize(vector<char> &buffer, vector<bh_base*> &data_send, vector<bh_base*> &data_recv)
{
    //Wrap 'buffer' in an io stream
    iostreams::basic_array_source<char> source(&buffer[0], buffer.size());
    iostreams::stream<iostreams::basic_array_source <char> > input_stream(source);
    archive::binary_iarchive ia(input_stream);

    //deserialize the BhIR
    bh_ir bhir;
    ia >> bhir;

    //Find all discarded base arrays (remote base pointers)
    for(const bh_instruction &instr: bhir.instr_list)
    {
        assert(instr.opcode >= 0);
        switch(instr.opcode)
        {
            case BH_DISCARD:
            {
                remote_discards.insert(instr.operand[0].base);
                break;
            }
            default: {}
        }
    }

    vector<bh_base> new_bases;//New base arrays in the order they appear in the instruction list
    ia >> new_bases;

    //Add the new base array to 'remote2local' and to 'data_recv'
    size_t new_base_count = 0;
    for(const bh_instruction &instr: bhir.instr_list)
    {
        const int nop = bh_noperands(instr.opcode);
        for(int i=0; i<nop; ++i)
        {
            const bh_view &v = instr.operand[i];
            if(bh_is_constant(&v))
                continue;
            if(remote2local.find(v.base) == remote2local.end())
            {
                assert(new_base_count < new_bases.size());
                remote2local[v.base] = new_bases[new_base_count++];
                if(remote2local[v.base].data != NULL)
                    data_recv.push_back(&remote2local[v.base]);
            }
        }
    }
    assert(new_base_count == new_bases.size());

    //Update all base pointers to point to the local bases
    for(bh_instruction &instr: bhir.instr_list)
    {
        const int nop = bh_noperands(instr.opcode);
        for(int i=0; i<nop; ++i)
        {
            bh_view &v = instr.operand[i];
            if(bh_is_constant(&v))
                continue;
            v.base = &remote2local[v.base];
        }
    }

    //Find base arrays that have data we must send
    for(const bh_instruction &instr: bhir.instr_list)
    {
        assert(instr.opcode >= 0);
        switch(instr.opcode)
        {
            case BH_SYNC:
            {
                data_send.push_back(instr.operand[0].base);
                break;
            }
            default: {}
        }
    }
    return bhir;
}

void ExecuteBackend::cleanup(const bh_ir &bhir)
{
    //Let's remove previously discarded base arrays (remote base pointers)
    for(const bh_base *base: remote_discards)
    {
        bh_data_free(&remote2local[base]);
        remote2local.erase(base);
    }
    remote_discards.clear();
}

}}
