/*
 * Copyright 2012 Andreas Thorning <thorning@diku.dk>
 *
 * This file is part of Bohrium.
 *
 * Bohrium is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Bohrium is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Bohrium. If not, see <http://www.gnu.org/licenses/>.
 */
 
#ifndef __USERFUNCTIONFFT_HPP
#define __USERFUNCTIONFFT_HPP

#include <string>
#include <map>
#include <bh.h>
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
    bh_error fft(bh_fft_type* fftDef, UserFuncArg* userFuncArg);
    bh_error fft2d(bh_fft_type* fftDef, UserFuncArg* userFuncArg);
};

#endif
