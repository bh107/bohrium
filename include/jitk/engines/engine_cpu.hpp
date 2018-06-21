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

#include "engine.hpp"

#include <bh_config_parser.hpp>
#include <jitk/statistics.hpp>
#include <jitk/apply_fusion.hpp>

#include <bh_view.hpp>
#include <bh_component.hpp>
#include <bh_instruction.hpp>
#include <bh_main_memory.hpp>

namespace bohrium {
namespace jitk {

class EngineCPU : public Engine {
public:
    EngineCPU(component::ComponentVE &comp, Statistics &stat) : Engine(comp, stat) {}

    ~EngineCPU() override = default;

    virtual void writeKernel(const LoopB &kernel,
                             const SymbolTable &symbols,
                             const std::vector<bh_base *> &kernel_temps,
                             uint64_t codegen_hash,
                             std::stringstream &ss) = 0;

    virtual void execute(const jitk::SymbolTable &symbols,
                         const std::string &source,
                         uint64_t codegen_hash,
                         const std::vector<const bh_instruction *> &constants) = 0;

    void handleExecution(BhIR *bhir) override;

    void handleExtmethod(BhIR *bhir) override;
};

}
} // namespace
