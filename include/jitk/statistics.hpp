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

#ifndef __BH_JITK_STATISTICS_H
#define __BH_JITK_STATISTICS_H

#include <chrono>
#include <string>
#include <ostream>
#include <sstream>
#include <vector>

#include <bh_instruction.hpp>

namespace bohrium {
namespace jitk {

namespace {
// Pretty print the ratio: 'a/b'
std::string pprint_ratio(uint64_t a, uint64_t b) {
    std::stringstream ss;
    ss << a << "/" << b << " (" << 100.0 * a / b << "%)";
    return ss.str();
}
}

/* BaseDB is a database over base arrays. The main feature is getBases(),
 * which always returns the bases in the order they where inserted.
 */
class Statistics {
  public:
    const bool enabled;
    uint64_t num_base_arrays=0;
    uint64_t num_temp_arrays=0;
    uint64_t num_syncs=0;
    uint64_t max_memory_usage=0;
    uint64_t totalwork=0;
    uint64_t threading_below_threshold=0;
    uint64_t kernel_cache_lookups=0;
    uint64_t kernel_cache_misses=0;
    uint64_t fuser_cache_lookups=0;
    uint64_t fuser_cache_misses=0;
    std::chrono::duration<double> time_total_execution{0};
    std::chrono::duration<double> time_fusion{0};
    std::chrono::duration<double> time_exec{0};
    std::chrono::duration<double> time_compile{0};
    std::chrono::duration<double> time_offload{0};
    std::chrono::duration<double> time_copy2dev{0};
    std::chrono::duration<double> time_copy2host{0};
    std::chrono::time_point<std::chrono::steady_clock> time_started{std::chrono::steady_clock::now()};

    Statistics(bool enabled) : enabled(enabled) {}

    // Pretty print the recorded statistics into 'out' where 'backend_name' is the name of the caller
    void pprint(std::string backend_name, std::ostream &out) {
        using namespace std;
        const chrono::duration<double> wallclock{chrono::steady_clock::now() - time_started};

        out << "[" << backend_name << "] Profiling: \n";
        out << "\tFuse Cache Hits:     " << pprint_ratio(fuser_cache_lookups - fuser_cache_misses, fuser_cache_lookups) << "\n";
        out << "\tKernel Cache Hits    " << pprint_ratio(kernel_cache_lookups - kernel_cache_misses, kernel_cache_lookups) << "\n";
        out << "\tArray contractions:  " << pprint_ratio(num_temp_arrays, num_base_arrays) << "\n";
        out << "\tMaximum Memory Usage: " << max_memory_usage / 1024 / 1024 << " MB\n";
        out << "\tSyncs to NumPy: " << num_syncs << "\n";
        out << "\tTotal Work: " << (double) totalwork << " operations\n";
        out << "\tWork below par-threshold(1000): " << threading_below_threshold / (double)totalwork * 100 << "%\n";
        out << "\tWall clock:  " << wallclock.count() << "s\n";
        out << "\tThroughput:  " << totalwork / (double)wallclock.count() << "ops\n";
        out << "\tTotal Execution:  " << time_total_execution.count() << "s\n";
        out << "\t  Fusion:    " << time_fusion.count() << "s\n";
        out << "\t  Compile:   " << time_compile.count() << "s\n";
        out << "\t  Exec:      " << time_exec.count() << "s\n";
        out << "\t  Copy2dev:  " << time_copy2dev.count() << "s\n";
        out << "\t  Copy2host: " << time_copy2host.count() << "s\n";
        out << "\t  Offload:   " << time_offload.count() << "s\n";
        out << endl;
    }

    // Record statistics based on the 'instr_list'
    void record(std::vector<bh_instruction> &instr_list) {
        if (enabled) {
            for (const bh_instruction &instr: instr_list) {
                if (not bh_opcode_is_system(instr.opcode)) {
                    const std::vector<bh_index> dshape = instr.dominating_shape();
                    totalwork += bh_nelements(dshape.size(), &dshape[0]);
                }
                if (instr.opcode == BH_SYNC) {
                    ++num_syncs;
                }
            }
        }
    }
};


} // jitk
} // bohrium

#endif
