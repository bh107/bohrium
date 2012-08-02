/*
 * Copyright 2012 Andreas Thorning <thorning@diku.dk>
 *
 * This file is part of cphVB.
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
 
#ifndef __USERFUNCTIONFFT_HPP
#define __USERFUNCTIONFFT_HPP

#include <string>
#include <map>
#include <cphvb.h>
#include "UserFuncArg.hpp"
#include "Kernel.hpp"
#include "StringHasher.hpp"


class UserFunctionFft
{
private:
    typedef std::map<std::string, Kernel> KernelMap;
    KernelMap kernelMap;
    ResourceManager* resourceManager;
public:
    UserFunctionFft(ResourceManager* rm);
    cphvb_error fft(cphvb_fft_type* fftDef, UserFuncArg* userFuncArg);
    cphvb_error fft2d(cphvb_fft_type* fftDef, UserFuncArg* userFuncArg);
};

#endif
