/*
This file is part of cphVB and copyright (c) 2012 the cphVB team:
http://cphvb.bitbucket.org

cphVB is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as 
published by the Free Software Foundation, either version 3 
of the License, or (at your option) any later version.

cphVB is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the 
GNU Lesser General Public License along with cphVB. 

If not, see <http://www.gnu.org/licenses/>.
*/

#include <vector>
#include <iostream>
#ifdef _WIN32
#include <sstream>
#endif
#include <cphvb.h>
#include "OCLtype.h"

void generateGIDSource(std::vector<cphvb_index> shape, std::ostream& source);
void generateOffsetSource(const cphvb_array* operand, std::ostream& source);
void generateInstructionSource(cphvb_opcode opcode,
                               OCLtype returnType, 
                               std::vector<std::string>& parameters, 
                               std::ostream& source);

