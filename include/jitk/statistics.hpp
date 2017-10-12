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
#include <fstream>
#include <vector>

#include <colors.hpp>
#include <bh_ir.hpp>
#include <bh_instruction.hpp>
#include <jitk/base_db.hpp>

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

class Statistics {
  public:
    bool enabled;
    bool print_on_exit; // On exist, write to file or pprint to stdout
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
    std::chrono::duration<double> time_total_execution{0};
    std::chrono::duration<double> time_pre_fusion{0};
    std::chrono::duration<double> time_fusion{0};
    std::chrono::duration<double> time_codegen{0};
    std::chrono::duration<double> time_compile{0};
    std::chrono::duration<double> time_exec{0};
    std::chrono::duration<double> time_offload{0};
    std::chrono::duration<double> time_copy2dev{0};
    std::chrono::duration<double> time_copy2host{0};
    std::chrono::duration<double> time_ext_method{0};

    std::chrono::duration<double> wallclock{0};
    std::chrono::time_point<std::chrono::steady_clock> time_started{std::chrono::steady_clock::now()};

    Statistics(bool enabled) : enabled(enabled), print_on_exit(enabled) {}
    Statistics(bool enabled, bool print_on_exit) : enabled(enabled), print_on_exit(print_on_exit) {}

    void write(std::string backend_name, std::string filename, std::ostream &out) {
        if (filename == "") {
            pprint(backend_name, out);
        } else {
            export_yaml(backend_name, filename);
        }
    }

    // Pretty print the recorded statistics into 'out' where 'backend_name' is the name of the caller
    void pprint(std::string backend_name, std::ostream &out) {
        using namespace std;

        if (enabled) {
            wallclock = chrono::steady_clock::now() - time_started;

            out << BLU << "[" << backend_name << "] Profiling: \n" << RST;
            out << "Fuse cache hits:                 " << GRN << fuse_cache_hits()                   << "\n" << RST;
            out << "Kernel cache hits                " << GRN << kernel_cache_hits()                 << "\n" << RST;
            out << "Array contractions:              " << GRN << array_contractions()                << "\n" << RST;
            out << "Outer-fusion ratio:              " << GRN << outer_fusion_ratio()                << "\n" << RST;
            out << "\n";
            out << "Max memory usage:                " << GRN << memory_usage() << " MB"             << "\n" << RST;
            out << "Syncs to NumPy:                  " << GRN << num_syncs                           << "\n" << RST;
            out << "Total Work:                      " << GRN << totalwork << " operations"          << "\n" << RST;
            out << "Throughput:                      " << GRN << throughput() << "ops"               << "\n" << RST;
            out << "Work below par-threshold (1000): " << GRN << work_below_thredshold() << "%"      << "\n" << RST;
            out << "\n";
            out << "Wall clock:                      " << BLU << wallclock.count() << "s"            << "\n" << RST;
            out << "Total Execution:                 " << BLU << time_total_execution.count() << "s" << "\n" << RST;
            out << "  Pre-fusion:                    " << YEL << time_pre_fusion.count() << "s"      << "\n" << RST;
            out << "  Fusion:                        " << YEL << time_fusion.count() << "s"          << "\n" << RST;
            out << "  Codegen:                       " << YEL << time_codegen.count() << "s"         << "\n" << RST;
            out << "  Compile:                       " << YEL << time_compile.count() << "s"         << "\n" << RST;
            out << "  Exec:                          " << YEL << time_exec.count() << "s"            << "\n" << RST;
            out << "  Copy2dev:                      " << YEL << time_copy2dev.count() << "s"        << "\n" << RST;
            out << "  Copy2host:                     " << YEL << time_copy2host.count() << "s"       << "\n" << RST;
            out << "  Ext-method:                    " << YEL << time_ext_method.count() << "s"      << "\n" << RST;
            out << "  Offload:                       " << YEL << time_offload.count() << "s"         << "\n" << RST;
            out << "  Other:                         " << YEL << time_other() << "s"                 << "\n" << RST;
            out << "\n";
            out << BOLD << RED << "Unaccounted for (wall - total):  " << unaccounted() << "s\n" << RST;
            out << endl;
        } else {
            out << BLU << "[" << backend_name << "] Profiling: " << RST;
            out << BOLD << RED << "Statistic Disabled\n" << RST;
        }
    }

    // Export statistic using the YAML format <http://yaml.org>
    void export_yaml(std::string backend_name, std::string filename) {
        using namespace std;

        if (enabled) {
            wallclock = chrono::steady_clock::now() - time_started;

            ofstream file;
            file.open(filename);

            file << "----"                                                      << "\n";
            file << backend_name << ":"                                         << "\n";
            file << "  fuse_cache_hits: "       << fuse_cache_hits()            << "\n";
            file << "  kernel_cache_hits: "     << kernel_cache_hits()          << "\n";
            file << "  array_contractions: "    << array_contractions()         << "\n";
            file << "  outer_fusion_ratio: "    << outer_fusion_ratio()         << "\n";
            file << "  memory_usage: "          << memory_usage()               << "\n"; // mb
            file << "  syncs: "                 << num_syncs                    << "\n";
            file << "  total_work: "            << totalwork                    << "\n"; // ops
            file << "  throughput: "            << throughput()                 << "\n"; // ops
            file << "  work_below_thredshold: " << work_below_thredshold()      << "\n"; // %
            file << "  timing:"                                                 << "\n";
            file << "    wall_clock: "          << wallclock.count()            << "\n"; // s
            file << "    total_execution: "     << time_total_execution.count() << "\n"; // s
            file << "    pre_fusion: "          << time_pre_fusion.count()      << "\n"; // s
            file << "    fusion: "              << time_fusion.count()          << "\n"; // s
            file << "    compile: "             << time_compile.count()         << "\n"; // s
            file << "    exec: "                << time_exec.count()            << "\n"; // s
            file << "    copy2dev: "            << time_copy2dev.count()        << "\n"; // s
            file << "    copy2host: "           << time_copy2host.count()       << "\n"; // s
            file << "    offload: "             << time_offload.count()         << "\n"; // s
            file << "    other: "               << time_other()                 << "\n"; // s
            file << "    unaccounted: "         << unaccounted()                << "\n"; // s

            file.close();
        }
    }

    // Record statistics based on the 'instr_list'
    void record(const BhIR &bhir) {
        if (enabled) {
            for (const bh_instruction &instr: bhir.instr_list) {
                if (instr.opcode != BH_IDENTITY and not bh_opcode_is_system(instr.opcode)) {
                    const std::vector<int64_t> shape = instr.shape();
                    totalwork += bh_nelements(shape.size(), &shape[0]);
                }
            }
            num_syncs += bhir.getSyncs().size();
        }
    }

    // Record statistics based on the 'symbols'
    void record(const SymbolTable& symbols) {
      num_base_arrays += symbols.getNumBaseArrays();
      num_temp_arrays += symbols.getNumBaseArrays() - symbols.getParams().size();
    }

  private:
    std::string fuse_cache_hits() {
        return pprint_ratio(fuser_cache_lookups - fuser_cache_misses, fuser_cache_lookups);
    }

    std::string kernel_cache_hits() {
        return pprint_ratio(kernel_cache_lookups - kernel_cache_misses, kernel_cache_lookups);
    }

    std::string array_contractions() {
        return pprint_ratio(num_temp_arrays, num_base_arrays);
    }

    std::string outer_fusion_ratio() {
        return pprint_ratio(num_blocks_out_of_fuser, num_instrs_into_fuser);
    }

    double memory_usage() {
        return (double) max_memory_usage / 1024.0 / 1024.0;
    }

    double throughput() {
        return (double) totalwork / (double) wallclock.count();
    }

    double work_below_thredshold() {
        return (double) threading_below_threshold / (double) totalwork * 100.0;
    }

    double time_other() {
        std::chrono::duration<double> time_other{0};
        return (time_total_execution - time_pre_fusion - time_fusion - time_codegen - time_compile - time_exec
                - time_copy2dev - time_copy2host - time_offload).count();
    }

    double unaccounted() {
        return (wallclock - time_total_execution).count();
    }
};

} // jitk
} // bohrium

#endif
