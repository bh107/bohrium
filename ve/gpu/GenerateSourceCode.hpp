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

#include <vector>
#include <iostream>
#ifdef _WIN32
#include <sstream>
#endif
#include <bh.h>
#include "OCLtype.h"

void generateGIDSource(size_t ndim, std::ostream& source);
void generateOffsetSource(size_t ndim, unsigned int id, std::ostream& source);
void generateInstructionSource(const bh_opcode opcode,
                               const std::vector<OCLtype>& type, 
                               const std::vector<std::string>& parameters,
                               const std::string& indent,
                               std::ostream& source);

