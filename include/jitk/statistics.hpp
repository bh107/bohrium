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
#include <colors.hpp>

namespace bohrium {
namespace jitk {
using namespace std;

class Statistics {
    public:

    const bool enabled;
    uint64_t num_base_arrays           = 0;
    uint64_t num_temp_arrays           = 0;
    uint64_t num_syncs                 = 0;
    uint64_t max_memory_usage          = 0;
    uint64_t totalwork                 = 0;
    uint64_t threading_below_threshold = 0;
    uint64_t kernel_cache_lookups      = 0;
    uint64_t kernel_cache_misses       = 0;
    uint64_t fuser_cache_lookups       = 0;
    uint64_t fuser_cache_misses        = 0;
    uint64_t num_instrs_into_fuser     = 0;
    uint64_t num_blocks_out_of_fuser   = 0;
    chrono::duration<double> time_total_execution{0};
    chrono::duration<double> time_pre_fusion{0};
    chrono::duration<double> time_fusion{0};
    chrono::duration<double> time_exec{0};
    chrono::duration<double> time_compile{0};
    chrono::duration<double> time_offload{0};
    chrono::duration<double> time_copy2dev{0};
    chrono::duration<double> time_copy2host{0};
    chrono::time_point<chrono::steady_clock> time_started{chrono::steady_clock::now()};

    Statistics(bool enabled) : enabled(enabled) {}

    // Pretty print the recorded statistics into 'out' where 'backend_name' is the name of the caller
    void pprint(string backend_name, ostream &out) {
        const chrono::duration<double> wallclock{chrono::steady_clock::now() - time_started};

        chrono::duration<double> time_other{0};
        time_other = time_total_execution - time_pre_fusion - time_fusion - time_compile - time_exec  - time_copy2dev - time_copy2host - time_offload;

        out << BLU << "[" << backend_name << "] Profiling: \n" << RST;
        out << "Fuse cache hits:                 " << GRN << pprint_ratio(fuser_cache_lookups - fuser_cache_misses, fuser_cache_lookups) << "\n" << RST;
        out << "Kernel cache hits                " << GRN << pprint_ratio(kernel_cache_lookups - kernel_cache_misses, kernel_cache_lookups) << "\n" << RST;
        out << "Array contractions:              " << GRN << pprint_ratio(num_temp_arrays, num_base_arrays) << "\n" << RST;
        out << "Outer-fusion ratio:              " << GRN << pprint_ratio(num_blocks_out_of_fuser, num_instrs_into_fuser) << "\n" << RST;
        out << "\n";
        out << "Max memory usage:                " << GRN << max_memory_usage / 1024 / 1024 << " MB\n" << RST;
        out << "Syncs to NumPy:                  " << GRN << num_syncs << "\n" << RST;
        out << "Total Work:                      " << GRN << (double) totalwork << " operations\n" << RST;
        out << "Throughput:                      " << GRN << totalwork / (double) wallclock.count() << "ops\n" << RST;
        out << "Work below par-threshold (1000): " << GRN << threading_below_threshold / (double) totalwork * 100 << "%\n" << RST;
        out << "\n";
        out << "Wall clock:                      " << BLU << wallclock.count() << "s\n" << RST;
        out << "Total Execution:                 " << BLU << time_total_execution.count() << "s\n" << RST;
        out << "  Pre-fusion:                    " << YEL << time_pre_fusion.count() << "s\n" << RST;
        out << "  Fusion:                        " << YEL << time_fusion.count() << "s\n" << RST;
        out << "  Compile:                       " << YEL << time_compile.count() << "s\n" << RST;
        out << "  Exec:                          " << YEL << time_exec.count() << "s\n" << RST;
        out << "  Copy2dev:                      " << YEL << time_copy2dev.count() << "s\n" << RST;
        out << "  Copy2host:                     " << YEL << time_copy2host.count() << "s\n" << RST;
        out << "  Offload:                       " << YEL << time_offload.count() << "s\n" << RST;
        out << "  Other:                         " << YEL << time_other.count() << "s\n" << RST;
        out << "\n";
        out << BOLD << RED << "Unaccounted for (wall-total):    " << (wallclock - time_total_execution).count() << "s\n" << RST;
        out << endl;
    }

    // Record statistics based on the 'instr_list'
    void record(vector<bh_instruction> &instr_list) {
        if (not enabled) {
            return;
        }

        for (const bh_instruction &instr: instr_list) {
            if (instr.opcode != BH_IDENTITY and not bh_opcode_is_system(instr.opcode)) {
                const vector<bh_index> shape = instr.shape();
                totalwork += bh_nelements(shape.size(), &shape[0]);
            }

            if (instr.opcode == BH_SYNC) {
                ++num_syncs;
            }
        }
    }

    private:
    // Pretty print the ratio: 'a/b'
    string pprint_ratio(uint64_t a, uint64_t b) {
        stringstream ss;
        ss << a << "/" << b << " (" << 100.0 * a / b << "%)";
        return ss.str();
    }
};

} // jitk
} // bohrium

#endif
