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

#ifndef __BH_DISPATCH_H
#define __BH_DISPATCH_H

#ifdef __cplusplus
extern "C" {
#endif

//Dispatch the instruction.
//NB: the instruction be a regular operation, i.e. no user-defined function, SYNC, etc.
bh_error dispatch(bh_instruction *instr);

//Dispatch the bundle of instructions.
//NB: the instruction be a regular operation, i.e. no user-defined function, SYNC, etc.
bh_error dispatch_bundle(bh_instruction** inst_bundle,
                            bh_intp size,
                            bh_intp nblocks);

//Initiate the dispather.
bh_error dispatch_init(void);

//Finalize the dispather.
bh_error dispatch_finalize(void);

#ifdef __cplusplus
}
#endif

#endif
