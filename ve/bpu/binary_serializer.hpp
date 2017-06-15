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

#ifndef __BINARY_SERIALIZER_HPP
#define __BINARY_SERIALIZER_HPP

#include <bh.h>

void dump_bhir(bh_ir* bhir);
void dump_memory_for_kernel(const bh_ir_kernel& kernel, const char* filename, const char* mapfilename);
void dump_instructions_for_kernel(const bh_ir_kernel& kernel, const char* filename);

#endif
