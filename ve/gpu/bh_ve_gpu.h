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
#ifndef __BH_VE_GPU_H
#define __BH_VE_GPU_H

#include <bh.h>
#include "InstructionScheduler.hpp"
#include "ResourceManager.hpp"
#include <bh_win.h>

#ifdef __cplusplus
extern "C" {
#endif

static bh_component* component = NULL;
static InstructionScheduler* instructionScheduler;
static ResourceManager* resourceManager;


DLLEXPORT bh_error bh_ve_gpu_init(bh_component* _component);
    
DLLEXPORT bh_error bh_ve_gpu_execute(bh_intp instruction_count,
                                           bh_instruction* instruction_list);

DLLEXPORT bh_error bh_ve_gpu_shutdown(void);

DLLEXPORT bh_error bh_ve_gpu_reg_func(char *fun, 
                                            bh_intp *id);

#ifdef __cplusplus
}
#endif

#endif
