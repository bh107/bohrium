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

#include <bh_component.hpp>

namespace bxx {

// Runtime : Definition
inline Runtime& Runtime::instance()
{
    static Runtime instance;
    return instance;
}

inline Runtime::Runtime() : global_random_seed_(0),
                            global_random_state_(0),
                            config(0), // stack level zero is the bridge
                            runtime(config.getChildLibraryPath(), 1), // and child is stack level 1
                            extension_count(BH_MAX_OPCODE_ID+1),
                            queue_size(0)
{
}

inline Runtime::~Runtime()
{
    if (ref_count.size()) {
        std::cout << "There are " << ref_count.size() << " dangling refs." << std::endl;
        for (std::map<bh_base*, size_t>::iterator it=ref_count.begin();
            it!=ref_count.end();
            ++it) {
            std::cout << it->first << " => " << it->second << '\n';
        }
    }
    flush();
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
 * Sends the instruction-list to the bohrium runtime.
 *
 * NOTE: Assumes that the list is non-empty.
 */
inline
size_t Runtime::execute()
{
    size_t cur_size = queue_size;

    bh_ir bhir = bh_ir(queue_size, queue);
    runtime.execute(&bhir);      // Send instructions to Bohrium
    queue_size = 0;                         // Reset size of the queue
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
 *  Create a non-linked operand without any meta-data.
 *  This is basically just "new multi_array<T>();"
 */
template <typename T>
inline
multi_array<T>& Runtime::create(void)
{
    multi_array<T>* operand = new multi_array<T>();
    return *operand;
}

/**
 * Create an operand based on another operands meta-data (shape)
 * But linked with its own bh_base.
 */
template <typename T, typename OtherT>
inline
multi_array<T>& Runtime::create_base(multi_array<OtherT>& input)
{
    multi_array<T>* operand = new multi_array<T>(input);
    operand->link();

    return *operand;
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
    Runtime::instance().ref_count[operand->meta.base] += 1;

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
    Runtime::instance().ref_count[operand->meta.base] += 1;

    return *operand;
}

template <typename TO, typename TL, typename TR>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<TO>& op0, multi_array<TL>& op1, multi_array<TR>& op2)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2] = op2.meta;

    if ((!identical(op0, op1)) && op0.getTemp()) { delete &op0; }
    if (op1.getTemp()) { delete &op1; }
    if (op2.getTemp()) { delete &op2; }
}

template <typename TO, typename TL, typename TR>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<TO>& op0, multi_array<TL>& op1, const TR op2)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2].base = NULL;
    assign_const_type(&instr->constant, op2);

    if ((!identical(op0, op1)) && op0.getTemp()) { delete &op0; }
    if (op1.getTemp()) { delete &op1; }
}

template <typename TO, typename TL, typename TR>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<TO>& op0, const TL op1, multi_array<TR>& op2)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1].base = NULL;
    instr->operand[2] = op2.meta;
    assign_const_type( &instr->constant, op1 );

    if ((!identical(op0, op2)) && op0.getTemp()) { delete &op0; }
    if (op2.getTemp()) { delete &op2; }
}

template <typename TO, typename TI>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<TO>& op0, multi_array<TI>& op1)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2].base = NULL;

    if ((!identical(op0, op1)) && op0.getTemp()) { delete &op0; }
    if (op1.getTemp()) { delete &op1; }
}

template <typename TO, typename TI>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<TO>& op0, const TI op1)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1].base = NULL;
    instr->operand[2].base = NULL;
    assign_const_type(&instr->constant, op1);

    if (op0.getTemp()) { delete &op0; }
}

template <typename TO>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<TO>& op0)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1].base = NULL;
    instr->operand[2].base = NULL;

    if (op0.getTemp()) { delete &op0; }
}

//
// This function is used to encode SYSTEM opcodes without operands (NONE and TALLY)
//
inline
void Runtime::enqueue(bh_opcode opcode)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0].base = NULL;
    instr->operand[1].base = NULL;
    instr->operand[2].base = NULL;
}

//
//  This function should only be used by random to encode the degenerate bh_r123 type.
//
template <typename TO>
inline
void Runtime::enqueue(bh_opcode opcode, multi_array<TO>& op0, const uint64_t op1, const uint64_t op2)
{
    bh_instruction* instr;

    guard();

    instr = &queue[queue_size++];
    instr->opcode = opcode;
    instr->operand[0] = op0.meta;
    instr->operand[1].base = NULL;
    instr->operand[2].base = NULL;

    instr->constant.type = BH_R123;
    instr->constant.value.r123.start = op1;
    instr->constant.value.r123.key   = op2;
}

template <typename T1, typename T2, typename T3>
inline
void Runtime::enqueue_extension(const std::string& name,
                                multi_array<T1>& op0,
                                multi_array<T2>& op1,
                                multi_array<T3>& op2)
{
    bh_instruction* instr;
    bh_opcode opcode;

    // Look for the extension opcode
    std::map<std::string, bh_opcode>::iterator it = extensions.find(name);
    if (it != extensions.end()) {   // Got it
        opcode = it->second;
    } else {                        // Add it
        opcode = extension_count++;
        extensions.insert(std::pair<std::string, bh_opcode>(name, opcode));
        runtime.extmethod(name.c_str(), opcode);
    }

    guard();

    // Construct and enqueue the instructions
    instr = &queue[queue_size++];
    instr->opcode = extensions[name];
    instr->operand[0] = op0.meta;
    instr->operand[1] = op1.meta;
    instr->operand[2] = op2.meta;

    if (op0.getTemp()) { delete &op0; }
    if (op1.getTemp()) { delete &op1; }
    if (op2.getTemp()) { delete &op2; }
}

template <typename T>
T scalar(multi_array<T>& op)
{
    bool is_temp = op.getTemp();
    op.setTemp(false);
    Runtime::instance().enqueue((bh_opcode)BH_SYNC, op);
    Runtime::instance().flush();

    bh_base *op_a = op.getBase();
    T* data = (T*)(op_a->data);
    data += op.meta.start;

    T value = *data;

    if (is_temp) {      // If it was a temp you will never see it again
        delete &op;     // so you better clean it up!
        Runtime::instance().flush();
    }

    return value;
}

inline void Runtime::trash(bh_base *base)
{
    garbage.push_back(base);
}

inline uint64_t Runtime::getRandSeed(void) {
    return global_random_seed_;
}

inline void Runtime::setRandSeed(uint64_t seed) {
    global_random_seed_ = seed;
}

inline uint64_t Runtime::getRandState(void) {
    return global_random_state_;
}

inline void Runtime::setRandState(uint64_t state) {
    global_random_state_ = state;
}

}
#endif

