#ifndef PROCESS_GRID_H
#define PROCESS_GRID_H
#include <mpi.h>
#include <cphvb.h>

#ifdef __cplusplus
extern "C" {
#endif

int myrank, worldsize;

/*===================================================================
 *
 * Initiate the MPI process grid.
 * NB: must be called before the use of myrank and worldsize
 */
void pgrid_init(void);


/*===================================================================
 *
 * Finalize the MPI process grid.
 */
void pgrid_finalize(void);


#ifdef __cplusplus
}
#endif

#endif
