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

#include <iostream>
#include <fstream>
#include <algorithm>
#include <stdexcept>
#include <bh.h>
#include <map>
#include <set>
#include <assert.h>

//#include <boost/archive/tmpdir.hpp>
//#include <boost/archive/xml_oarchive.hpp>
//#include <boost/archive/binary_oarchive.hpp>
//#include <boost/archive/text_oarchive.hpp>

#include "binary_serializer.hpp"

#define OUTPUT_TAG "[BPU-VE] "

#define NULLBUFFERSIZE (8*1024)
static char NULLBUFFER[NULLBUFFERSIZE];


// Add serialization to bh_ir_kernel
/*namespace boost {
    namespace serialization {
        template<class Archive>
            void serialize(Archive & ar, bh_ir_kernel & kernel, const unsigned int version)
            {
                ar & kernel.get_bases();
                ar & kernel.get_temps();
                ar & kernel.get_frees();
                ar & kernel.get_discards();
                ar & kernel.get_syncs();
                ar & kernel.get_parameters();

                ar & kernel.get_output_map();
                ar & kernel.get_output_set();
                ar & kernel.get_input_map();
                ar & kernel.get_input_set();
                ar & kernel.get_input_shape();

                ar & kernel.is_scalar(); 
                ar & kernel.get_sweeps();

                ar & kernel.get_constants();

                ar & kernel.instr_indexes();
            }

        }
}*/

static int bpu_kernel_count=1;

void dump_bhir(bh_ir* bhir)
{
    char filename[8000];
    char filename2[8000];

    int i = 1;

    for(std::vector<bh_ir_kernel>::iterator kernel = bhir->kernel_list.begin(); kernel != bhir->kernel_list.end(); ++kernel)
    {
        std::cout << OUTPUT_TAG << "Processing kernel " << i << std::endl;

    	snprintf(filename, 8000, "mem-%d-%d.bin", bpu_kernel_count, i);
    	snprintf(filename2, 8000, "map-%d-%d.bin", bpu_kernel_count, i);
    	dump_memory_for_kernel(*kernel, filename, filename2);
    	snprintf(filename, 8000, "instr-%d-%d.bin", bpu_kernel_count, i);
    	dump_instructions_for_kernel(*kernel, filename);

    	i++;
    }

    bpu_kernel_count++;

}

void dump_memory_for_kernel(const bh_ir_kernel& kernel, const char* filename, const char* mapfilename)
{
    std::set<bh_base*> bases = kernel.get_bases();
    std::map<bh_base*, unsigned long> memory_offsets;

    unsigned long mempos = 0;

    std::ofstream memfile(filename, std::ios::binary | std::ios::trunc | std::ios::out);
    assert(memfile.good());

    for(bh_base* cb: bases)
    {
        memory_offsets.insert(std::pair<bh_base*, unsigned long>(cb, mempos));
        bh_index datasize = bh_base_size(cb);

        if (cb->data == NULL)
        {
            bh_index remain = datasize;
            while(remain > 0)
            {
                memfile.write(NULLBUFFER, std::min(remain, (bh_index)NULLBUFFERSIZE));
                remain -= std::min(remain, (bh_index)NULLBUFFERSIZE);
            }
        }
        else
        {
            memfile.write((char*)(cb->data), bh_base_size(cb));
        }

        mempos += datasize;

        assert(memfile.tellp() == mempos);
    }

    std::cout << OUTPUT_TAG << " dumped memory file with " << mempos << " bytes" << std::endl;

    std::ofstream mapfile(mapfilename, std::ios::binary | std::ios::trunc | std::ios::out);
    assert(mapfile.good());
	size_t map_count = memory_offsets.size();
    mapfile.write((char*)&map_count, sizeof(size_t));

    for(auto& kv : memory_offsets)
    {
	    mapfile.write((char*)&(kv.first), sizeof(bh_base*));
	    mapfile.write((char*)&(kv.second), sizeof(unsigned long));
    }
}
    

void dump_instructions_for_kernel(const bh_ir_kernel& kernel, const char* filename)
{
	const std::vector<uint64_t> instructions = kernel.instr_indexes();

    std::ofstream instrfile(filename, std::ios::binary | std::ios::trunc | std::ios::out);
    assert(instrfile.good());

	size_t instruction_count = instructions.size();
    instrfile.write((char*)&instruction_count, sizeof(size_t));

    for(auto& instr_index : instructions)
    {
        bh_instruction instr = kernel.bhir->instr_list[instr_index];

		instrfile.write((char*)&instr.opcode, sizeof(bh_opcode));
		instrfile.write((char*)&instr.constant, sizeof(bh_constant));

		unsigned int operands = bh_operands_in_instruction(&instr);
		instrfile.write((char*)&operands, sizeof(unsigned int));
		for(int i = 0; i < operands; i++)
		{
			bh_view view = instr.operand[i];

			instrfile.write((char*)&view.base, sizeof(bh_base*));
			if (view.base != NULL)
			{
				instrfile.write((char*)&view.base->data, sizeof(bh_data_ptr));
				instrfile.write((char*)&view.base->type, sizeof(bh_type));
				instrfile.write((char*)&view.base->nelem, sizeof(bh_index));
			}
            else
            {
                //instrfile.write((char*)&instr.constant, sizeof(bh_constant));
            }

			instrfile.write((char*)&view.start, sizeof(bh_index));
			
			bh_intp dimensions = view.ndim;
			instrfile.write((char*)&dimensions, sizeof(bh_intp));

			for(int j = 0; j < dimensions; j++)
			{
				instrfile.write((char*)&view.shape[j], sizeof(bh_index));
				instrfile.write((char*)&view.stride[j], sizeof(bh_index));				
			}
		}
    }
}