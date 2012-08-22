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

#ifndef __CPHVB_BUNDLE_H
#define __CPHVB_BUNDLE_H

#ifdef __cplusplus
extern "C" {
#endif

/* Calculates the bundleable instructions.
 *
 * @inst The instruction list
 * @size Size of the instruction list
 * @return Number of consecutive bundeable instruction
 */
cphvb_intp cphvb_inst_bundle(cphvb_instruction* insts, cphvb_intp start, cphvb_intp end);

#ifdef __cplusplus
}
#endif

#endif
