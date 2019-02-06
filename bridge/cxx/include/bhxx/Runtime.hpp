/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

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

#include <iostream>
#include <sstream>

#include "BhInstruction.hpp"
#include <bh_component.hpp>

namespace bhxx {

/**
 *  Encapsulation of communication with Bohrium runtime.
 *  Implemented as a Singleton.
 *
 *  \note  Not thread-safe.
 */
class Runtime {
public:
    Runtime();

    ~Runtime() {
        flush();
    }

    /// Get the singleton instance of the Runtime class
    static Runtime &instance() {
        static Runtime instance;
        return instance;
    }

    /// Create and enqueue a new bh_instruction based on `opcode` and a variadic
    /// pack of BhArrays and at most one scalar value
    template<typename T, typename... Ts>
    void enqueue(bh_opcode opcode, T &op, Ts &... ops);

    /** Enqueue any BhInstruction object */
    void enqueue(BhInstruction instr);

    /// We have to handle random specially because of the `BH_R123` scalar type
    void enqueueRandom(BhArray<uint64_t> &out, uint64_t seed, uint64_t key);

    /// Enqueue an extension method
    template<typename T>
    void enqueueExtmethod(const std::string &name, BhArray<T> &out, BhArray<T> &in1,
                          BhArray<T> &in2);

    /** Schedule a base object for deletion
     *
     * Will call BH_FREE on it first at the next flush
     */
    void enqueueDeletion(std::unique_ptr<BhBase> base_ptr);

    /// Send enqueued instructions to Bohrium for execution
    void flush();

    /** Flush and repeat the lazy evaluated operations until `base_ptr` is false or `nrepeats` is reached
     *
     * @param nrepeats  Number of maximum repeats
     * @param base_ptr  Repeat while `base_ptr` is true or is null. NB: must be an array with one element of type BH_BOOL
     */
    void flushAndRepeat(uint64_t nrepeats, const std::shared_ptr<BhBase> &base_ptr);

    /// Flag array to be sync'ed after the next flush
    void sync(const std::shared_ptr<BhBase> &base_ptr);

    /** Changes the offset and shape of a view between the iterations of a `do_while` loop.
     * This is the underlying functionality behind using iterators.
     *
     * @param view_ptr     A pointer to the view on which the changes are applied
     * @param dim          The dimension of the changes
     * @param slide        The change to the offset (can be both positive and negative)
     * @param shape_change The change to the shape (can be both positive and negative)
     * @param view_shape   The shape of the view (necessary when using negative indexing)
     * @param view_stride  The stride used for changes to the offset
     * @param step_delay   The delay of the changes (used for nested loops)
     */
    template <typename T>
    void slide_view(BhArray<T>* view_ptr, int64_t dim, int64_t slide, int64_t shape_change,
                    int64_t view_shape, int64_t view_stride, int64_t step_delay) {
        view_ptr->slides().dims.push_back({dim, slide, shape_change, view_stride, view_shape, step_delay});
    }

    /** Resets the changes made to the offset/shape of a dimension.
     * This is the underlying functionality behind iterator grid (nested loops).
     * Used within `do_while` to emulate nested loops.
     *
     * @param view_ptr A pointer to the view which is reset
     * @param dim      The dimension of the reset
     * @param reset_it The amount of iterations before a reset
     */
    template <typename T>
    void add_reset(BhArray<T>* view_ptr, int64_t dim, int64_t reset_it) {
        view_ptr->slides().resets[dim] = std::make_pair(reset_it, 0);
    }

    /// Send and receive a message through the component stack
    std::string message(const std::string &msg);

    /** Get data pointer from the first VE in the runtime stack
     * NB: this doesn't include a flush.
     *
     * @param base         The base array that owns the data for retrieval
     * @param copy2host    Always copy the memory to main memory
     * @param force_alloc  Force memory allocation
     * @param nullify      Set the data pointer to NULL after returning
     * @return       The data pointer (NB: might point to device memory)
     * Throws exceptions on error
     */
    void *getMemoryPointer(std::shared_ptr<BhBase> &base, bool copy2host, bool force_alloc, bool nullify);

    /** Set data pointer in the first VE in the runtime stack
     * NB: The component will deallocate the memory when encountering a BH_FREE.
     *     Also, this doesn't include a flush
     *
     * @param base      The base array that will own the data
     * @param host_ptr  The pointer points to the host memory (main memory) as opposed to device memory
     * @param mem       The data pointer
     * Throws exceptions on error
     */
    void setMemoryPointer(std::shared_ptr<BhBase> &base, bool host_ptr, void *mem);

    /** Copy the memory of `src` to `dst`
     *
     * @tparam T     The type of the arrays
     * @param src    The source
     * @param dst    The destination
     * @param param  Parameters to compression (use the empty string for no compression)
     */
    template <typename T>
    void memCopy(BhArray<T> &src, BhArray<T> &dst, const std::string &param) {
        bh_view _src = src.getBhView();
        bh_view _dst = dst.getBhView();
        runtime.memCopy(_src, _dst, param);
    }

    /** Get the device handle, such as OpenCL's cl_context, of the first VE in the runtime stack.
     * If the first VE isn't a device, NULL is returned.
     *
     * @return  The device handle
     * Throws exceptions on error
     */
    void *getDeviceContext();

    /** Set the device context, such as CUDA's context, of the first VE in the runtime stack.
     * If the first VE isn't a device, nothing happens
     *
     * @param device_context  The new device context
     * Throws exceptions on error
     */
    void setDeviceContext(void *device_context);

    /// Get the number of calls to flush so far
    uint64_t getFlushCount() { return _flush_count; }

    /** Run a user kernel
     *
     * @param kernel The source code of the kernel
     * @param operand_list The operands given to the kernel all of which must be regular arrays
     * @param compile_cmd The compilation command
     * @param tag String that define which component that should handle this kernel (e.g. "openmp", "opencl", or "cuda")
     * @param param Backend specific parameters (e.g. OpenCL needs `global_work_size` and `local_work_size`)
     * @return The compiler output (both stdout and stderr) when the compilation fails else it is the empty string
     */
    std::string userKernel(const std::string &kernel, std::vector<BhArrayUnTypedCore*> &operand_list,
                           const std::string &compile_cmd, const std::string &tag, const std::string &param);

private:
    //@{
    /** BH_FREE for arrays is special, since we deal with the deletion of the
     * base implictly via the BhBaseDeleter (which in turn calls
     * enqueueDeletion in this object).
     *
     * This function just resets the shared pointers of the array, which
     * might trigger the call of enqueueDeletion, but only if the
     * array is really no longer needed.
     * */
    template<typename T>
    void freeMemory(BhArray<T> &ary);
    //@}

    // The lazy evaluated instructions
    std::vector<bh_instruction> instr_list;

    // The base arrays to sync when flushing
    std::set<bh_base *> syncs;

    // Unique pointers to base objects, which are to be
    // purged after the next flush
    std::vector<std::unique_ptr<BhBase> > bases_for_deletion;

    // Bohrium Configuration
    bohrium::ConfigParser config;

    // The Bohrium Runtime i.e. the child of this component
    bohrium::component::ComponentFace runtime;

    // Mapping an extension method name to an opcode id
    std::map<std::string, bh_opcode> extmethods;

    // The opcode id for the next new extension method
    bh_opcode extmethod_next_opcode_id;

    // Number of calls to flush
    uint64_t _flush_count = 0;
};

//
// ----------------------------------------------------------
//

template<typename T, typename... Ts>
void Runtime::enqueue(bh_opcode opcode, T &op, Ts &... ops) {
    if (opcode == BH_FREE) {
        // BH_FREE is special, see the freeMemory function why.
        assert(sizeof...(Ts) == 0);
        freeMemory(op);
    } else {
        BhInstruction instr(opcode);
        instr.appendOperand(op, ops...);
        enqueue(std::move(instr));
    }
}

template<typename T>
void Runtime::enqueueExtmethod(const std::string &name, BhArray<T> &out, BhArray<T> &in1,
                               BhArray<T> &in2) {
    bh_opcode opcode;

    // Look for the extension opcode
    auto it = extmethods.find(name);
    if (it != extmethods.end()) {
        opcode = it->second;
    } else {
        // Add it and tell rest of Bohrium about this new extmethod
        opcode = extmethod_next_opcode_id++;
        runtime.extmethod(name.c_str(), opcode);
        extmethods.insert(std::pair<std::string, bh_opcode>(name, opcode));
    }

    // Now that we have an opcode, let's enqueue the instruction
    enqueue(opcode, out, in1, in2);
}

template<typename T>
void Runtime::freeMemory(BhArray<T> &ary) {
    // Calling BH_FREE on an array with external
    // storage management is undefined behaviour
    if (!ary.base()->ownMemory()) {
        throw std::runtime_error(
                "Cannot call BH_FREE on a BhArray object, which uses external storage "
                "in its BhBase.");
    }

    // BH_FREE is special because it is automatically invoked
    // by the deleter of the shared pointer to BhBase if the last
    // array referencing BhBase is deleted.
    // So instead of actually deleting anything we will just
    // remove our reference to the BhBase instead
    ary.base().reset();
}
}
