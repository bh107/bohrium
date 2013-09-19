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
#ifndef __BOHRIUM_BRIDGE_CPP_RUNTIME
#define __BOHRIUM_BRIDGE_CPP_RUNTIME
#include <iostream>
#include <sstream>

namespace bh {

// Runtime : Definition
inline Runtime& Runtime::instance()
{
    static Runtime instance;
    return instance;
}

inline Runtime::Runtime() : random_id(0), ext_in_queue(0), queue_size(0)
{
    bh_error err;
    char err_msg[100];

    int64_t        component_count; // Bohrium Runtime / Bridge setup
    bh_component **components;          

    bridge = bh_component_setup(NULL);
    bh_component_children(bridge, &component_count, &components);

    if (component_count != 1 || (!((components[0]->type == BH_VEM) || \
                                   (components[0]->type != BH_VEM)))) {
        sprintf(err_msg, "Error in the runtime configuration: the bridge must "
                         "have exactly one child of type VEM or FILTER.\n");
        free(components);
        throw std::runtime_error(err_msg);
    }
    runtime = components[0];
    free(components);

    err = runtime->init(runtime);   // Initialize child
    if (err) {
        fprintf(stderr, "Error in runtime->init(runtime)\n");
        exit(-1);
    }

    //
    // Register extensions
    //
    err = runtime->reg_func("bh_random", &random_id);
    if (err != BH_SUCCESS) {
        sprintf(err_msg, "Fatal error in the initialization of the user"
                        "-defined random operation: %s.\n",
                        bh_error_text(err));
        throw std::runtime_error(err_msg);
    }
    if (random_id <= 0) {
        sprintf(err_msg, "Fatal error in the initialization of the user"
                        "-defined random operation: invalid ID returned"
                        " (%ld).\n", (long)random_id);
        throw std::runtime_error(err_msg);
    }
}

inline Runtime::~Runtime()
{
    flush();

    runtime->shutdown();
    bh_component_free(runtime);
    bh_component_free(bridge);
}

inline size_t Runtime::get_queue_size()
{
    return queue_size;
}

/**
 * De-allocate all bh_base associated with BH_DISCARD instruction in the queue.
 *
 * @param count The max number of instructions to look at.
 * @return The number of bh_base that got de-allocated.
 */
inline
size_t Runtime::deallocate_meta(size_t count)
{
    size_t deallocated = 0;
    if (count > BH_CPP_QUEUE_MAX) {
        throw std::runtime_error("Trying to de-allocate more than physically possible.");
    }
    while(!garbage.empty()) {
        delete garbage.front();
        ++deallocated;
        garbage.pop_front();
    }
    return deallocated;
}

/**
 * De-allocate all meta-data associated with any USERFUNCs in the instruction queue.
 *
 * @return The number of USERFUNCs that got de-allocated.
 */
inline
size_t Runtime::deallocate_ext()
{
    size_t deallocated = 0;

    if (ext_in_queue>0) {
        for(size_t i=0; i<ext_in_queue; i++) {
            if (ext_queue[i]->id == random_id) {
                free((bh_random_type*)ext_queue[i]);
                deallocated++;
            } else {
                throw std::runtime_error("Cannot de-allocate extension...");
            }
        }
        ext_in_queue = 0;
    }
    return deallocated;
}

/**
 * Sends the instruction-list to the bohrium runtime.
 *
 * NOTE: Assumes that the list is non-empty.
 */
inline
size_t Runtime::execute()
{
    size_t cur_size = queue_size;
    
    bh_ir bhir;
    bh_error status = bh_ir_create(&bhir, queue_size, queue);
    if (status == BH_SUCCESS) {
        status = runtime->execute(&bhir);   // Send instructions to Bohrium
        queue_size = 0;                     // Reset size of the queue
    }
    bh_ir_destroy(&bhir);

    if (status != BH_SUCCESS) {
        std::stringstream err_msg;
        err_msg << "Err: Runtime::execute() child->execute() failed: " << bh_error_text(status) << std::endl;

        throw std::runtime_error(err_msg.str());
    }

    deallocate_ext();
    deallocate_meta(cur_size);

    return cur_size;
}

/**
 * Flush the instruction-queue if it is about to get overflowed.
 *
 * This is used as a pre-condition to adding instructions to the queue.
 *
 * @return The number of instructions flushed.
 */
inline
size_t Runtime::guard()
{
    if (queue_size >= BH_CPP_QUEUE_MAX) {
        return execute();
    }
    return 0;
}

/**
 * Flush the instruction-queue.
 *
 * This will force the runtime system to execute the queued up instructions,
 * at least if there are any queued instructions.
 *
 * @return The number of instructions flushed.
 */
inline
size_t Runtime::flush()
{
    if (queue_size > 0) {
        return execute();
    }
    return 0;
}

/**
 * Create an unitialized intermediate operand.
 */
template <typename T>
inline
multi_array<T>& Runtime::temp()
{
    multi_array<T>* operand = new multi_array<T>();
    operand->setTemp(true);

    return *operand;
}

/**
 * Create an intermediate operand based on another operands meta-data.
 */
template <typename T, typename OtherT>
inline
multi_array<T>& Runtime::temp(multi_array<OtherT>& input)
{
    multi_array<T>* operand = new multi_array<T>(input);
    operand->setTemp(true);

    return *operand;
}

/**
 * Create an alias/segment/view of the supplied base operand.
 */
template <typename T>
inline
multi_array<T>& Runtime::view(multi_array<T>& base)
{
    multi_array<T>* operand = new multi_array<T>(base);
    operand->meta.base = base.meta.base;

    return *operand;
}

/**
 * Create an intermediate alias/segment/view of the supplied base operand.
 */
template <typename T>
inline
multi_array<T>& Runtime::temp_view(multi_array<T>& base)
{
    multi_array<T>* operand = new multi_array<T>(base);
    operand->meta.base = base.meta.base;
    operand->setTemp(true);

    return *operand;
}

template <typename T>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<T>& op0, multi_array<T>& op1, multi_array<T>& op2)
{
    bh_instruction* instr;

    guard();
    
    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2] = op2.meta;

    if (op1.getTemp()) { delete &op1; }
    if (op2.getTemp()) { delete &op2; }
}

template <typename T>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<T>& op0, multi_array<T>& op1, const T& op2)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2].base = NULL;
    assign_const_type(&instr->constant, op2);

    if (op1.getTemp()) { delete &op1; }
}

template <typename T>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<T>& op0, const T& op1, multi_array<T>& op2)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1].base = NULL;
    instr->operand[2] = op2.meta;
    assign_const_type( &instr->constant, op1 );

    if (op2.getTemp()) { delete &op2; }
}

template <typename T>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<T>& op0, multi_array<T>& op1)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2].base = NULL;

    if (op1.getTemp()) { delete &op1; }
}

template <typename T>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<T>& op0, const T& op1)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1].base = NULL;
    instr->operand[2].base = NULL;
    assign_const_type(&instr->constant, op1);
}

template <typename T>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<T>& op0)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1].base = NULL;
    instr->operand[2].base = NULL;
}

template <typename Ret, typename In>    // x = y
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<Ret>& op0, multi_array<In>& op1)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2].base = NULL;

    if (op1.getTemp()) { delete &op1; }
}

template <typename Ret, typename In>    // x = y < z
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<Ret>& op0, multi_array<In>& op1, multi_array<In>& op2)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2] = op2.meta;
    assign_const_type( &instr->constant, op2 );

    if (op1.getTemp()) { delete &op1; }
    if (op2.getTemp()) { delete &op2; }
}

template <typename Ret, typename In>    // x = y < 1
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<Ret>& op0, multi_array<In>& op1, const In& op2)
{
    bh_instruction* instr;

    guard();
    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2].base = NULL;
    assign_const_type( &instr->constant, op2 );

    if (op1.getTemp()) { delete &op1; }
}

template <typename Ret, typename In>    // reduce(), pow()
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<Ret>& op0, multi_array<Ret>& op1, const In& op2)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2].base = NULL;
    assign_const_type( &instr->constant, op2 );

    if (op1.getTemp()) { delete &op1; }
}

template <typename Ret, typename In>    // x = 1 < y
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<Ret>& op0, const In& op1, multi_array<In>& op2)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1].base = NULL;
    instr->operand[2] = op2.meta;
    assign_const_type( &instr->constant, op1 );

    if (op2.getTemp()) { delete &op2; }
}

template <typename T>           // Userfunc / extensions
inline
void Runtime::enqueue(bh_userfunc* rinstr)
{
    bh_instruction* instr;

    guard();   

    instr = &queue[queue_size++];
    instr->opcode        = BH_USERFUNC;
    instr->userfunc      = (bh_userfunc *) rinstr;

    ext_queue[ext_in_queue++] = instr->userfunc;
}

template <typename T>
T scalar(multi_array<T>& op)
{
    Runtime::instance().enqueue((bh_opcode)BH_SYNC, op);
    Runtime::instance().flush();

    bh_base *op_a = op.getBase();
    T* data = (T*)(op_a->data);
    data += op.meta.start;

    T value = *data;

    if (op.getTemp()) { // If it was a temp you will never see it again
        delete &op;     // so you better clean it up!
        Runtime::instance().flush();
    }

    return value;
}

inline void Runtime::trash(bh_base *base_ptr)
{
    garbage.push_back(base_ptr);
}

}
#endif

