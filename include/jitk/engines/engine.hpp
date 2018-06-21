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

#include <bh_config_parser.hpp>
#include <jitk/statistics.hpp>
#include <jitk/instruction.hpp>
#include <jitk/view.hpp>
#include <jitk/fuser_cache.hpp>
#include <jitk/codegen_cache.hpp>

#include <bh_view.hpp>
#include <bh_component.hpp>
#include <bh_instruction.hpp>
#include <boost/filesystem.hpp>

namespace bohrium {
namespace jitk {

/** The base class of a Engine, which is the component that transforms bytecode into machine code.
 * All Vector Engines to inherent from this class
 */
class Engine {
protected:
    component::ComponentVE &comp;
    Statistics &stat;
    FuseCache fcache;
    CodegenCache codegen_cache;
    const bool verbose;

    // Maximum number of cache files
    const int64_t cache_file_max;

    // Path to a temporary directory for the source and object files
    const boost::filesystem::path tmp_dir;

    // Path to the temporary directory of the source files
    const boost::filesystem::path tmp_src_dir;

    // Path to the temporary directory of the binary files (e.g. .so files)
    const boost::filesystem::path tmp_bin_dir;

    // Path to the directory of the cached binary files (e.g. .so files)
    const boost::filesystem::path cache_bin_dir;

    // The hash of the JIT compilation command
    uint64_t compilation_hash{0};

    // The malloc cache limit in percent and bytes.
    // NB: each backend should set and use these values with the malloc cache
    int64_t malloc_cache_limit_in_percent{-1};
    int64_t malloc_cache_limit_in_bytes{-1};

public:
    /** The only constructor */
    Engine(component::ComponentVE &comp, Statistics &stat) :
            comp(comp),
            stat(stat),
            fcache(stat),
            codegen_cache(stat),
            verbose(comp.config.defaultGet<bool>("verbose", false)),
            cache_file_max(comp.config.defaultGet<int64_t>("cache_file_max", 50000)),
            tmp_dir(get_tmp_path(comp.config)),
            tmp_src_dir(tmp_dir / "src"),
            tmp_bin_dir(tmp_dir / "obj"),
            cache_bin_dir(comp.config.defaultGet<boost::filesystem::path>("cache_dir", "")),
            compilation_hash(0) {
        // Let's make sure that the directories exist
        jitk::create_directories(tmp_src_dir);
        jitk::create_directories(tmp_bin_dir);
        if (not cache_bin_dir.empty()) {
            jitk::create_directories(cache_bin_dir);
        }
    }

    virtual ~Engine() = default;

    /** Return general information of the engine (should be human readable) */
    virtual std::string info() const = 0;

    /** Return the type of `dtype` as a string (e.g. uint64_t or cl_int64)
     *
     * @param dtype The data type
     * @return The string with the written type
     */
    virtual const std::string writeType(bh_type dtype) = 0;

    /** Set the `bh_instruction->constructor` flag of all instruction in `instr_list`
     * The constructor flag indicates whether the instruction construct the output array
     * (i.e. is the first operation on that array)
     *
     * @param instr_list         The list of instruction to update
     * @param constructed_arrays Arrays already constructed. Will be updated with arrays constructed in `instr_list`
     */
    virtual void setConstructorFlag(std::vector<bh_instruction *> &instr_list, std::set<bh_base *> &constructed_arrays);

    /** Set the `bh_instruction->constructor` flag of all instruction in `instr_list`
     * The constructor flag indicates whether the instruction construct the output array
     * (i.e. is the first operation on that array)
     *
     * @param instr_list  The list of instruction to update
     */
    virtual void setConstructorFlag(std::vector<bh_instruction *> &instr_list) {
        std::set<bh_base *> constructed_arrays;
        setConstructorFlag(instr_list, constructed_arrays);
    };

    /** Update statistics with final aggregated values of the engine */
    virtual void updateFinalStatistics() {} // Default we do nothing

protected:

    /** Handle execution of the `bhir` */
    virtual void handleExecution(BhIR *bhir) = 0;

    /** Handle extension methods in the `bhir` */
    virtual void handleExtmethod(BhIR *bhir) = 0;

    /** Write the argument list of the kernel function, which is basicly a comma seperated list of arguments.
     *
     * @param symbols           The symbol table
     * @param ss                The stream output
     * @param array_type_prefix If not null, a string to prepend each argument
     */
    virtual void writeKernelFunctionArguments(const jitk::SymbolTable &symbols,
                                              std::stringstream &ss,
                                              const char *array_type_prefix);

    /** Writes a kernel, which corresponds to a set of for-loop nest.
     *
     * @param symbols       The symbol table
     * @param parent_scope  The callers scope object or null when there is no parant
     * @param kernel        The kernel (LoopB block with rank -1) to write
     * @param thread_stack  A vector that specifies the amount of parallelism in each nest level (excl. rank -1)
     * @param opencl        Is this a OpenCL/CUDA kernel?
     * @param out           The stream output
     */
    virtual void writeBlock(const SymbolTable &symbols,
                            const Scope *parent_scope,
                            const LoopB &kernel,
                            const std::vector<uint64_t> &thread_stack,
                            bool opencl,
                            std::stringstream &out);

    /** Write a loop header
     *
     * @param symbols       The symbol table
     * @param scope         The scope
     * @param block         The block
     * @param thread_stack  A vector that specifies the amount of parallelism in each nest level (excl. rank -1)
     * @param out           The stream output
     */
    virtual void loopHeadWriter(const SymbolTable &symbols,
                                Scope &scope,
                                const LoopB &block,
                                const std::vector<uint64_t> &thread_stack,
                                std::stringstream &out) = 0;
};

}
} // namespace
