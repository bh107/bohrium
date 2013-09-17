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
#include <bh.h>
#include <set>

#include "transitive_reduction.h"

/* Performing a transitive reduction on all DAGs in the bhir using the O(n^3)
 * method introduced by Aho, Garey & Ullman (1972).
 * <http://epubs.siam.org/doi/abs/10.1137/0201008>
 */
void transitive_reduction_filter(bh_ir *bhir)
{
    for(bh_intp d=0; d<bhir->ndag; ++d)
    {
        bh_dag *dag = &bhir->dag_list[d];
        bh_adjmat *adjmat = &dag->adjmat;
        bh_intp nnode = dag->nnode;

        //Find redundant dependencies.
        //NB: will include dependencies that dosn't exist.
        std::set<bh_intp> redundant[nnode];
        for(bh_intp k=0; k<nnode; ++k)
        {
            bh_intp row_size, col_size;
            const bh_intp *row = bh_adjmat_get_row(adjmat, k, &row_size);
            const bh_intp *col = bh_adjmat_get_col(adjmat, k, &col_size);
            for(bh_intp c=0; c<col_size; ++c)
                for(bh_intp r=0; r<row_size; ++r)
                    redundant[col[c]].insert(row[r]);
        }

        //Lets create a new copy of the rows in the adjmat
        bh_boolmat old_boolmat = adjmat->m;
        bh_error e = bh_boolmat_create(&adjmat->m, nnode);
        if(e != BH_SUCCESS)
        {
            printf("The creation of the boolean matrix failed: %s\n", bh_error_text(e));
            throw std::exception();
        }

        //Fill the new adjmat -- one row at a time.
        for(bh_intp k=0; k<nnode; ++k)
        {
            bh_intp row_size;
            //First we remove all redundant dependencies from the row
            const bh_intp *row = bh_boolmat_get_row(&old_boolmat, k, &row_size);
            bh_intp new_row[row_size], size=0;
            for(bh_intp r=0; r<row_size; ++r)
            {
                if(redundant[k].erase(row[r]) == 0)
                    new_row[size++] = row[r];
            }
            e = bh_boolmat_fill_empty_row(&adjmat->m, k, size, new_row);
            if(e != BH_SUCCESS)
            {
                printf("Filling of row %ld in the boolean matrix failed: %s\n", k, bh_error_text(e));
                throw std::exception();
            }
            e = bh_boolmat_transpose(&adjmat->mT, &adjmat->m);
            if(e != BH_SUCCESS)
            {
                printf("Transposing the boolean matrix failed: %s\n", bh_error_text(e));
                throw std::exception();
            }
        }
    }
}



