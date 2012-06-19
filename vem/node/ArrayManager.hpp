/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __ARRAYMANAGER_HPP
#define __ARRAYMANAGER_HPP

#include <map>
#include <deque>
#include <cphvb.h>
#include <StaticStore.hpp>

typedef struct
{
    cphvb_bool opcode[CPHVB_NO_OPCODES];//list of opcode support
    cphvb_bool type[CPHVB_NO_OPCODES];  //list of type support
} cphvb_support;


/* Codes for known components */
typedef enum
{
    CPHVB_PARENT,
    CPHVB_SELF,
    CPHVB_CHILD

} owner_t;

struct OwnerTicket
{
    cphvb_instruction* instruction;
    cphvb_array* array;
    owner_t owner;
};

class ArrayManager
{
private:
    StaticStore<cphvb_array>* arrayStore;
    std::deque<cphvb_instruction*> eraseQueue;
    std::deque<cphvb_instruction*> freeQueue;
    std::deque<OwnerTicket> ownerChangeQueue;
    void erase(cphvb_array* base);

public:
    ArrayManager();
    ~ArrayManager();
    cphvb_array* create(cphvb_array* base,
                        cphvb_type type,
                        cphvb_intp ndim,
                        cphvb_index start,
                        cphvb_index shape[CPHVB_MAXDIM],
                        cphvb_index stride[CPHVB_MAXDIM]);
    void erasePending(cphvb_instruction* inst);
    void freePending(cphvb_instruction* inst);
    void changeOwnerPending(cphvb_instruction* inst, 
                        cphvb_array* base,
                        owner_t owner);
    void flush();
};


#endif
