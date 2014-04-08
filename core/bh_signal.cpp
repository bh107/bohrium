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


#include "bh_signal.h"
#include <pthread.h>
#define PAGE_SIZE getpagesize()

#define PAGE_ALIGN(address) ((uintptr_t) (((uintptr_t)address) & (~((uintptr_t)(PAGE_SIZE-1)))))


struct mSpace
{
    unsigned long idx;
    uintptr_t start;
    uintptr_t end;
    void (*callback)(unsigned long, uintptr_t);
};
pthread_mutex_t mspace_mutex = PTHREAD_MUTEX_INITIALIZER;
static long mspaceid;
static int init;
static int spacesize;
static int spacesmax;
static int used;

static mSpace *mspaces;
static long *idssorted;
static uintptr_t *ptssorted;

static void sighandler(int signal_number, siginfo_t *info, void *context);

static void expandarrays(void);

static int addspace(mSpace m);
static int removespace(int idx);

void quicksort(uintptr_t arr[], int low, int high, long idarr[]);

int search(uintptr_t start);
static int searchinterval(uintptr_t start, uintptr_t end);
static int binarysearch(uintptr_t value, uintptr_t arr[], int high);

static int findid(signed long id, void (*callback));


void print_spaces()
{
    printf("MSPACES |");
    for (int i =0; i < spacesize; i+=2)
    {
        mSpace c = mspaces[idssorted[i]];
        printf("%p <-> %p|", (void*)c.start, (void*)c.end);
    }
    printf("\n");
}
/** Signal handler.
 *  Executes appropriate callback function associated with memory segment.
 *
 * @param signal_number The signal number for SIGSEGV
 * @param siginfo_t Datastructure containing signal information.
 * @param context User context for the signal trap.
 * @return void
 */
static void sighandler(int signal_number,
                       siginfo_t *info,
                       void *context)
{
    //signal(signal_number, SIG_IGN);
    pthread_mutex_lock(&mspace_mutex);
    int memarea = search((uintptr_t)info->si_addr);
    //printf("SEGF Address: %p\n", info->si_addr);
    //print_spaces();
    if (memarea == -1)
    {
        pthread_mutex_unlock(&mspace_mutex);
        printf("bh_signal: Defaulting to segfaul at addr: %p\n", info->si_addr);
        signal(signal_number, SIG_DFL);
    }
    else
    {
        mSpace m = mspaces[memarea];
        pthread_mutex_unlock(&mspace_mutex);
        m.callback(m.idx, (uintptr_t)info->si_addr);
    }
}


/** Expand internal arrays to hold a more elements.
 *
 * @param void
 * @return void
 */
static void expandarrays(void)
{
    spacesmax = spacesmax * 2;
    idssorted = (long int *)realloc(idssorted, (sizeof(long)*spacesmax)*2);
    ptssorted = (uintptr_t *)realloc(ptssorted, sizeof(uintptr_t)*spacesmax*2);
    mspaces = (mSpace *)realloc(mspaces, sizeof(mSpace)*spacesmax);
}

/** Init arrays and signal handler
 *
 * @param void
 * @returnm void
 */
int init_signal(void){
    struct sigaction sact;

    sigfillset(&(sact.sa_mask));
    sact.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sact.sa_sigaction = sighandler;
    sigaction(SIGSEGV, &sact, &sact);
    if (init == 0)
    {
        mspaceid = 0;
        spacesize = 0;
        spacesmax = 5;

        idssorted = (long *)malloc(spacesmax * sizeof(long) * 2);
        ptssorted = (uintptr_t *)malloc(spacesmax * sizeof(uintptr_t) * 2);
        mspaces = (mSpace *)malloc(spacesmax * sizeof(mSpace));
    }
    init = 1;
    return 0;
}


/** Attach continues memory segment to signal handler
 *
 * @param idx - Id to identify the memory segment when executing the callback function.
 * @param start - Start address of memory segment.
 * @param size - Size of memory segment in bytes
 * @param callback - Callback function, executed when segfault hits in the memory segment.
 * @return - error code
 */
int attach_signal(signed long idx, // id to execute call back function with
                  uintptr_t start, // start address of memory segment
                  long int size,  // size in bytes
                  void (*callback)(unsigned long, uintptr_t))
{
    // Check if the memory area is already present
    pthread_mutex_lock(&mspace_mutex);
    uintptr_t end = start + size - 1;
    if (searchinterval(start, end) != -1)
    {
        pthread_mutex_unlock(&mspace_mutex);
        printf("Could not attach signal, memory segment is in conflict with already attached signal\n");
        return -1;
    }
    if (mspaceid == spacesmax -1)
    {
        expandarrays();
    }
    // Add area to mspaces
    mSpace m = {idx, start, end, callback};
    //printf("Attaching Signal (%p, %p)\n", (void*)start, (void*)end);
    int ret = addspace(m);
    pthread_mutex_unlock(&mspace_mutex);
    // Setup mprotect for the area
    //if (mprotect((void *)start, size, PROT_NONE) == -1)
    //{
    //    int errsv = errno;
    //    printf("Could not not mprotect array, error: %s.\n", strerror(errsv));
    //    return -1;
    //}

    //print_spaces();
    return 0;
}


/** Detach signal
 *
 * @param id - Segment id identifies the memory segment.
 * @param callback - Callback function as provided with the attach function.
 * @return - error code
 */
int detach_signal(signed long id, void (*callback)){
    pthread_mutex_lock(&mspace_mutex);
    int idx = findid(id, callback);
    if (idx == -1)
    {
        pthread_mutex_unlock(&mspace_mutex);
    //    printf("Could not detach signal with unique id (%li, %p)\n", id, callback);
        return -1;
    }
    removespace(idx);
    pthread_mutex_unlock(&mspace_mutex);
    return 0;
}


/** Remove memory segment from internal datastructures
 *
 * @param id - Internal segment id.
 * @return - error code
 */
static int removespace(int id)
{
    mSpace m = mspaces[id];
    int midx = binarysearch(m.start, ptssorted, spacesize-1);
    //printf("Detaching Signal: (%p, %p), %i\n", (void*)m.start, (void *)m.end, midx);
    if ((midx % 2) == 0)
    {
        printf("Could not remove signal, wrong signal id, %i.\n", id);
        return -1;
    }
    ptssorted[midx] = ptssorted[spacesize-1]+1;
    ptssorted[midx-1] = ptssorted[spacesize-1]+1;
    quicksort(&ptssorted[0], 0, spacesize-1, idssorted);
    spacesize -= 2;
    return 0;
}


/** Remove memory segment from internal datastructures
 *
 * @param id - Internal segment id.
 * @return - error code
 */
static int findid(signed long idx, void (*callback))
{
    int i;
    for (i=0; i < spacesize; i+=2)
    {
        int pos = idssorted[i];
        if (mspaces[pos].callback == callback && mspaces[pos].idx == idx)
        {
            return pos;
        }
    }
    return -1;
}


/** Search the internal datastructures for a given memory interval
 *
 * @param start - Pointer to start of memory segment.
 * @param end - Pointer to end of memory segment.
 * @return - error code.
 */
static int searchinterval(uintptr_t start, uintptr_t end)
{

    if (start > end || spacesize == 0)
        return -1;
    int startbound = binarysearch(start, ptssorted, spacesize-1);
    int endbound = binarysearch(end, ptssorted, spacesize-1);
    if (startbound == endbound)
    {
        if (startbound % 2)
            return idssorted[startbound];
        else
            return -1;
    }
    else
        return idssorted[startbound];

    return -1;
}


/** Add a mSpace type to the internal data structures
 *
 * @param m - mSpace typedef, internal struct holding a mem segment.
 * @return - error code.
 */
static int addspace(mSpace m)
{
    mspaces[mspaceid] = m;
    idssorted[spacesize] = mspaceid;
    ptssorted[spacesize] = m.start;
    idssorted[spacesize+1] = mspaceid;
    ptssorted[spacesize+1] = m.end;

    mspaceid += 1;
    spacesize += 2;

    quicksort(&ptssorted[0], 0, spacesize-1, idssorted);

    return mspaceid;
}


/** Add a mSpace type to the internal data structures
 *
 * @param m - mSpace typedef, internal struct holding a mem segment.
 * @return - error code.
 */
int search(uintptr_t value)
{
    int val = binarysearch(value, ptssorted, spacesize-1);
    if (val % 2)
        return idssorted[val];

    return -1;
}


/** Binary search
 *
 * @param value - Value to search the arrays containing pointers
 * @param arr[] - Array of pointer pairs.
 * @param high - End of array.
 * @return - error code.
 */
static int binarysearch(uintptr_t value, uintptr_t arr[], int high)
{
    int low = 0;
    while (low <= high)
    {
        int mid = (low + high) / 2;
        if (arr[mid] > value)
            high = mid - 1;
        else
            low = mid + 1;
    }
    return low;
}


/** Quick sort
 *
 * @param value - Value to search the arrays containing pointers
 * @param arr[] - Array of pointer pairs.
 * @param high - Size of array.
 * @param idarr - Internal id array.
 * @return - error code.
 */
void quicksort(uintptr_t arr[], int low, int high, long idarr[])
{
    int i = low;
    int j = high;
    uintptr_t y = 0;
    long x = 0;

    /* compare value */
    uintptr_t z = arr[(low + high) / 2];
    /* partition */
    do
    {
        /* find member above ... */
        while(arr[i] < z) i++;

        /* find element below ... */
        while(arr[j] > z) j--;

        if(i <= j)
        {
            /* swap two elements */
            y = arr[i];
            arr[i] = arr[j];
            arr[j] = y;
            x = idarr[i];
            idarr[i] = idarr[j];
            idarr[j] = x;

            i++;
            j--;
        }
    }
    while(i <= j);

    /* recurse */
    if(low < j)
        quicksort(arr, low, j, idarr);

    if(i < high)
        quicksort(arr, i, high, idarr);
}
