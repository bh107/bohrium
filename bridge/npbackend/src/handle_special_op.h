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

#pragma once

#include "util.h"

/** Handle extension methods
 *
 * @param name          The name of the extension method
 * @param operand_list  List of operands that can be NumPy-arrays and  Bohrium-arrays but NOT Scalars
 *                      NB: all dtype must be identical.
 */
PyObject *PyExtMethod(PyObject *self, PyObject *args, PyObject *kwds);

/** Execute the delayed instruction */
PyObject* PyFlush(PyObject *self, PyObject *args);

/** Get the number of times flush has been called */
PyObject* PyFlushCount(PyObject *self, PyObject *args);

/** Flush and repeat the lazy evaluated operations while `condition` is true and `nrepeats` hasn't been reach
 *
 * @param nrepeats   Maximum number of times to repeat the lazy evaluated operations
 * @param condition  The finish condition
 * */
PyObject* PyFlushCountAndRepeat(PyObject *self, PyObject *args);

/** Sync `ary` to host memory */
PyObject* PySync(PyObject *self, PyObject *args, PyObject *kwds);

/** Increases `ary`s offset by one */
PyObject* PySlideView(PyObject *self, PyObject *args, PyObject *kwds);

/** Add a reset for a given dimension. */
PyObject* PyAddReset(PyObject *self, PyObject *args, PyObject *kwds);

/** Create a new flat random array using the random123 algorithm.
    The dtype is uint64 always.

    @param size  The number of elements in the new flat array
    @param seed  The seed of a random sequence
    @param key   The index in the random sequence
*/
PyObject* PyRandom123(PyObject *self, PyObject *args, PyObject *kwds);

/** Return a pointer to the bhc data of `ary`
 *
 * @param ary          The bharray in question
 * @param copy2host    When true, always copy the data to main memory before returning
 * @param force_alloc  When true, force memory allocation before returning the data pointer
 * @param nullify      When true, set the bhc array's data NULL after returning the data pointer
 * @return             The data pointer of the bhc array
 */
void *get_data_pointer(BhArray *ary, bhc_bool copy2host, bhc_bool force_alloc, bhc_bool nullify);

/** Return a pointer to the bhc data of `ary`
 *  This is the same as `get_data_pointer()` but returns the pointer as a Python integer
 */
PyObject* PyGetDataPointer(PyObject *self, PyObject *args, PyObject *kwds);

/** Set the data pointer of `ary`
 *  NB: The data will be deallocate when the bhc array is freed
 *
 * @param ary       The bharray in question
 * @param mem_ptr   The new data pointer given as a Python integer, which is casted to (void *)
 * @param host_ptr  When true, the pointer points to the host memory (main memory) as opposed to device memory
 */
PyObject* PySetDataPointer(PyObject *self, PyObject *args, PyObject *kwds);

/** Copy the memory of `src` to `dst`
 *
 * @param src    The source array
 * @param dst    The destination array
 * @param param  the Parameters to compression (use the empty string for no compression)
 */
PyObject* PyMemCopy(PyObject *self, PyObject *args, PyObject *kwds);

/** Get the device context, such as OpenCL's cl_context, of the first VE in the runtime stack */
PyObject* PyGetDeviceContext(PyObject *self, PyObject *args);

/** Send and receive a message through the Bohrium stack
 *
 * @param msg  The message to send down to the bhc runtime
 * @return     The message answer from the bhc runtime
 */
PyObject* PyMessage(PyObject *self, PyObject *args, PyObject *kwds);

/** Run a user kernel
*
* @param kernel The source code of the kernel
* @param operand_list The operands given to the kernel all of which must be regular arrays (not scalars)
* @param compile_cmd The compilation command
* @return The compiler output (both stdout and stderr) when the compilation fails else it is the empty string
*/
PyObject* PyUserKernel(PyObject *self, PyObject *args, PyObject *kwds);