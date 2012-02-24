/*
 * Copyright 2012 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __KERNEL_HPP
#define __KERNEL_HPP

#include <CL/cl.hpp>
#include <vector>
#include <cphvb.h>
#include "KernelArg.hpp"
#include "ResourceManager.hpp"

class Kernel
{
private:
    ResourceManager* resourceManager;
    cphvb_intp ndim;
    std::vector<OCLtype> signature;
    cl::Kernel kernel;
public:
    Kernel(ResourceManager* resourceManager_, 
           cphvb_intp ndim_,
           const std::vector<OCLtype>& signature_,
           const std::string& source, 
           const std::string& name); 
    void call(const std::vector<KernelArg>& args, 
              const std::vector<cphvb_index>& shape);
};

#endif
