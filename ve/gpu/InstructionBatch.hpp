/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
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

#ifndef __INSTRUCTIONBATCH_HPP
#define __INSTRUCTIONBATCH_HPP

#include <vector>
#include <map>
#include <set>
#include <cphvb.h>
#include "BaseArray.hpp"

class InstructionBatch
{
    typedef std::map<cphvb_array*, std::pair<BaseArray*,std::string>> ArrayMap;
    typedef std::map<cphvb_array*, std::string> ScalarMap;
    typedef std::map<BaseArray*, cphvb_array*> OutputMap;
    typedef std::multimap<BaseArray*, cphvb_array*> InputMap;
private:
    std::vector<cphvb_index> shape;
    std::vector<cphvb_instruction*> instructions;
    OutputMap output;
    InputMap input;
    ArrayMap arrayParameters;
    ScalarMap scalarParameters;
    ScalarMap kernelVariables;
    int arraynum;
    int scalarnum;
    int variablenum;
    static int kernel;
    bool shapeMatch(cphvb_intp ndim, const cphvb_index dims[]);
    bool sameView(const cphvb_array* a, const cphvb_array* b);
    void generateInstructionSource(cphvb_opcode opcode, 
                                   std::vector<std::string>& parameters, 
                                   std::ostream& source);
    void generateOffsetSource(cphvb_array* operand, std::ostream& source);
public:
    InstructionBatch(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase);
    std::string generateCode();
    void add(cphvb_instruction* inst, const std::vector<BaseArray*>& operandBase);
    bool read(BaseArray*);
    bool write(BaseArray*);    
    bool access(BaseArray*);
};

class BatchException 
{
private:
    int code;
public:
    BatchException(int code_) : code(code_) {}
};


#endif
