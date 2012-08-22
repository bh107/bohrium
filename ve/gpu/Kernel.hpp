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

#ifndef __KERNEL_HPP
#define __KERNEL_HPP

#include <CL/cl.hpp>
#include <vector>
#include <cphvb.h>
#include "ResourceManager.hpp"
#include "BaseArray.hpp"

class Kernel
{
private:
    ResourceManager* resourceManager;
    cphvb_intp ndim;
    cl::Kernel kernel;
public:
    typedef std::vector<std::pair<KernelParameter*, bool> > Parameters;
    Kernel(ResourceManager* resourceManager_, 
           cphvb_intp ndim_,
           cl::Kernel kernel_);
    Kernel(ResourceManager* resourceManager_, 
           cphvb_intp ndim_,
           const std::string& source, 
           const std::string& name); 
    void call(Parameters& parameters,
              const std::vector<size_t>& globalShape);
    void call(Parameters& parameters,
              const std::vector<size_t>& globalShape,
              const std::vector<size_t>& localShape);
    static std::vector<Kernel> createKernels(ResourceManager* resourceManager_, 
                                             const std::vector<cphvb_intp> ndims,
                                             const std::string& source, 
                                             const std::vector<std::string>& kernelNames); 
    static  std::vector<Kernel> createKernelsFromFile(ResourceManager* resourceManager_, 
                                                      const std::vector<cphvb_intp> ndims,
                                                      const std::string& fileName, 
                                                      const std::vector<std::string>& kernelNames); 
};

#endif
