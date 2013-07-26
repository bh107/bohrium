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

#ifndef __BH_ADJMAT_H
#define __BH_ADJMAT_H

#include <bh.h>
#include "bh_boolmat.h"

#ifdef __cplusplus
extern "C" {
#endif


/* The adjacency matrix (bh_adjmat) represents edges between
 * nodes <http://en.wikipedia.org/wiki/Adjacency_matrix>.
 * The adjacencies are directed such that a row index represents
 * the source node and the column index represents the target node.
 * In this implementation, we use sparse boolean matrices to store
 * the adjacencies but alternatives such as adjacency lists or
 * incidence lists is also a possibility.
*/
typedef struct
{
    //Boolean matrix with the adjancencies
    bh_boolmat m;
} bh_adjmat;



#ifdef __cplusplus
}
#endif

#endif

