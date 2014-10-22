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

#ifndef __KERNEL_HPP
#define __KERNEL_HPP

#include "cl.hpp"
#include <vector>
#include <bh.h>
#include "bh_ve_gpu.h"
#include "ResourceManager.hpp"
#include "BaseArray.hpp"

class Kernel
{
private:
    bh_intp ndim;
    cl::Kernel kernel;
public:
    typedef std::vector<std::pair<KernelParameter*, bool> > Parameters;
    Kernel(cl::Kernel kernel_);
    Kernel(const std::string& source, 
           const std::string& name,
           const std::string& options = std::string("")); 
    void call(Parameters parameters,
              const std::vector<size_t> globalShape);
    void call(Parameters parameters,
              const std::vector<size_t> globalShape,
              const std::vector<size_t> localShape);
    static std::vector<Kernel> createKernels(const std::string& source, 
                                             const std::vector<std::string>& kernelNames,
                                             const std::string& options = std::string("")); 
    static  std::vector<Kernel> createKernelsFromFile(const std::string& fileName, 
                                                      const std::vector<std::string>& kernelNames,
                                                      const std::string& options = std::string("")); 
};

#endif
