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

#ifndef __CPHVB_DISPATCH_H
#define __CPHVB_DISPATCH_H

#ifdef __cplusplus
extern "C" {
#endif

//Dispatch the instruction.
//NB: the instruction be a regular operation, i.e. no user-defined function, SYNC, etc.
cphvb_error dispatch(cphvb_instruction *instr);

//Dispatch the bundle of instructions.
//NB: the instruction be a regular operation, i.e. no user-defined function, SYNC, etc.
cphvb_error dispatch_bundle(cphvb_instruction** inst_bundle,
                            cphvb_intp size,
                            cphvb_intp nblocks);

//Initiate the dispather.
cphvb_error dispatch_init(void);

//Finalize the dispather.
cphvb_error dispatch_finalize(void);

#ifdef __cplusplus
}
#endif

#endif
