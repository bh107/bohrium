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

#include <chrono>
#include <string>
#include <ostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <vector>

#include <colors.hpp>
#include <bh_ir.hpp>
#include <bh_instruction.hpp>
#include <bh_config_parser.hpp>
#include <jitk/symbol_table.hpp>
#include <jitk/codegen_util.hpp>

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

struct KernelStats {
  uint64_t num_calls = 0;
  std::chrono::duration<double> total_time{0};
  std::chrono::duration<double> max_time{0};
  std::chrono::duration<double> min_time{std::numeric_limits<double>::infinity()};

  bool operator< (const KernelStats& rhs) const {
    // default ordering: by total time
    return this->total_time.count() < rhs.total_time.count();
  }

  void register_exec_time(const std::chrono::duration<double>& exec_time) {
    ++num_calls;
    total_time += exec_time;
    max_time = max(max_time, exec_time);
    min_time = min(min_time, exec_time);
  }
};

class Statistics {
  public:
    bool enabled;
    bool print_on_exit; // On exist, write to file or pprint to stdout
    bool verbose; // Print per-kernel statistics
    uint64_t num_base_arrays           = 0;
    uint64_t num_temp_arrays           = 0;
    uint64_t num_syncs                 = 0;
    uint64_t max_memory_usage          = 0;
    uint64_t totalwork                 = 0;
    uint64_t threading_below_threshold = 0;
    uint64_t fuser_cache_lookups       = 0;
    uint64_t fuser_cache_misses        = 0;
    uint64_t codegen_cache_lookups     = 0;
    uint64_t codegen_cache_misses      = 0;
    uint64_t kernel_cache_lookups      = 0;
    uint64_t kernel_cache_misses       = 0;
    uint64_t num_instrs_into_fuser     = 0;
    uint64_t num_blocks_out_of_fuser   = 0;
    uint64_t malloc_cache_lookups      = 0;
    uint64_t malloc_cache_misses       = 0;
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

    // key: kernel source filename, value: kernel statistics
    std::map<std::string, KernelStats> time_per_kernel;

    std::chrono::duration<double> wallclock{0};
    std::chrono::time_point<std::chrono::steady_clock> time_started{std::chrono::steady_clock::now()};

    Statistics(const ConfigParser &config) : enabled(config.defaultGet("prof", false)),
                                             print_on_exit(config.defaultGet("prof", false)),
                                             verbose(config.defaultGet("verbose", false)) {}
    Statistics(bool enabled, const ConfigParser &config) : enabled(enabled),
                                                           print_on_exit(config.defaultGet("prof", false)),
                                                           verbose(config.defaultGet("verbose", false)) {}

    void write(std::string backend_name, std::string filename, std::ostream &out) {
        if (filename == "") {
            pprint(backend_name, out);
        } else {
            exportYAML(backend_name, filename);
        }
    }

    // Pretty print the recorded statistics into 'out' where 'backend_name' is the name of the caller
    void pprint(std::string backend_name, std::ostream &out) {
        using namespace std;

        if (enabled) {
            wallclock = chrono::steady_clock::now() - time_started;

            out << BLU << "[" << backend_name << "] Profiling: \n" << RST;
            out << "Fuse cache hits:                 " << GRN << fuseCacheHits()                     << "\n" << RST;
            out << "Codegen cache hits:              " << GRN << codegenCacheHits()                  << "\n" << RST;
            out << "Compilation cache hits:          " << GRN << kernelCacheHits()                   << "\n" << RST;
            out << "Array contractions:              " << GRN << arrayContractions()                 << "\n" << RST;
            out << "Outer-fusion ratio:              " << GRN << outerFusionRatio()                  << "\n" << RST;
            out << "Malloc cache hits:               " << GRN << MallocCacheHits()                   << "\n" << RST;
            out << "\n";
            out << "Max memory usage:                " << GRN << memoryUsage() << " MB"              << "\n" << RST;
            out << "Syncs to NumPy:                  " << GRN << num_syncs                           << "\n" << RST;
            out << "Total Work:                      " << GRN << totalwork << " operations"          << "\n" << RST;
            out << "Throughput:                      " << GRN << throughput() << "ops"               << "\n" << RST;
            out << "Work below par-threshold (1000): " << GRN << workBelowThredshold() << "%"        << "\n" << RST;
            out << "\n";
            out << "Wall clock:                      " << BLU << wallclock.count() << "s"            << "\n" << RST;
            out << "Total Execution:                 " << BLU << time_total_execution.count() << "s" << "\n" << RST;
            out << "  Pre-fusion:                    " << YEL << time_pre_fusion.count() << "s"      << "\n" << RST;
            out << "  Fusion:                        " << YEL << time_fusion.count() << "s"          << "\n" << RST;
            out << "  Codegen:                       " << YEL << time_codegen.count() << "s"         << "\n" << RST;
            out << "  Compilation:                   " << YEL << time_compile.count() << "s"         << "\n" << RST;
            out << "  Exec:                          " << YEL << time_exec.count() << "s"            << "\n" << RST;
            out << "  Copy2dev:                      " << YEL << time_copy2dev.count() << "s"        << "\n" << RST;
            out << "  Copy2host:                     " << YEL << time_copy2host.count() << "s"       << "\n" << RST;
            out << "  Offload:                       " << YEL << time_offload.count() << "s"         << "\n" << RST;
            out << "  Other:                         " << YEL << timeOther() << "s"                  << "\n" << RST;
            out << "Ext-method:                      " << YEL << time_ext_method.count() << "s"      << "\n" << RST;
            out << "\n";
            out << BOLD << RED << "Unaccounted for (wall - total):  " << unaccounted() << "s\n" << RST;

            if (verbose) {
              out << "\n";
              out << BLU << "Per-kernel Profiling:"                                                  << "\n" << RST;
              out << "  " << std::left << std::setw(39) << "Kernel filename"
                                       << std::setw(14) << "Calls"
                                       << std::setw(12) << "Total time"
                                       << std::setw(12) << "Max time"
                                       << std::setw(12) << "Min time"                                << "\n" << RST;
              auto cmp = [](std::pair<std::string, KernelStats> const & a, std::pair<std::string, KernelStats> const & b) {
                // compare map by values (descending)
                return !(a.second < b.second);
              };
              std::vector<std::pair<std::string, KernelStats> > tpk_sorted(time_per_kernel.begin(), time_per_kernel.end());
              std::sort(std::begin(tpk_sorted), std::end(tpk_sorted), cmp);
              for (auto const& x : tpk_sorted) {
                std::string kernel_filename = x.first;
                KernelStats kernel_data = x.second;
                out << "  "
                    << std::left         << std::setw(39) << kernel_filename
                    << std::right << YEL << std::setw(10) << kernel_data.num_calls         << "    "
                    << std::scientific   << std::setprecision(2)
                                         << std::setw(8) << kernel_data.total_time.count() << "s   "
                                         << std::setw(8) << kernel_data.max_time.count()   << "s   "
                                         << std::setw(8) << kernel_data.min_time.count()   << "s   " << "\n" << RST;
              }
            }
            out << endl;
        } else {
            out << BLU << "[" << backend_name << "] Profiling: " << RST;
            out << BOLD << RED << "Statistic Disabled\n" << RST;
        }
    }

    // Export statistic using the YAML format <http://yaml.org>
    void exportYAML(std::string backend_name, std::string filename) {
        using namespace std;

        if (enabled) {
            wallclock = chrono::steady_clock::now() - time_started;

            ofstream file;
            file.open(filename);

            file << "----"                                                           << "\n";
            file << backend_name << ":"                                              << "\n";
            file << "  fuse_cache_hits: "       << fuseCacheHits()                   << "\n";
            file << "  codegen_cache_hits: "    << codegenCacheHits()                << "\n";
            file << "  kernel_cache_hits: "     << kernelCacheHits()                 << "\n";
            file << "  array_contractions: "    << arrayContractions()               << "\n";
            file << "  outer_fusion_ratio: "    << outerFusionRatio()                << "\n";
            file << "  memory_usage: "          << memoryUsage()                     << "\n"; // mb
            file << "  syncs: "                 << num_syncs                         << "\n";
            file << "  total_work: "            << totalwork                         << "\n"; // ops
            file << "  throughput: "            << throughput()                      << "\n"; // ops
            file << "  work_below_thredshold: " << workBelowThredshold()             << "\n"; // %
            file << "  timing:"                                                      << "\n";
            file << "    wall_clock: "          << wallclock.count()                 << "\n"; // s
            file << "    total_execution: "     << time_total_execution.count()      << "\n"; // s
            file << "    pre_fusion: "          << time_pre_fusion.count()           << "\n"; // s
            file << "    fusion: "              << time_fusion.count()               << "\n"; // s
            file << "    compile: "             << time_compile.count()              << "\n"; // s
            file << "    exec: "                                                     << "\n";
            file << "      total: "             << time_exec.count()                 << "\n"; // s
            if (verbose) {
              file << "      per_kernel: "                                           << "\n";
              for (auto const& x : time_per_kernel) {
                KernelStats kernel_data = x.second;
                file << "        - " << x.first << ": "                              << "\n";
                file << "            num_calls: "  << kernel_data.num_calls          << "\n";
                file << "            total_time: " << kernel_data.total_time.count() << "\n"; // s
                file << "            max_time: "   << kernel_data.max_time.count()   << "\n"; // s
                file << "            min_time: "   << kernel_data.min_time.count()   << "\n"; // s
              }
            }
            file << "    copy2dev: "            << time_copy2dev.count()             << "\n"; // s
            file << "    copy2host: "           << time_copy2host.count()            << "\n"; // s
            file << "    offload: "             << time_offload.count()              << "\n"; // s
            file << "    other: "               << timeOther()                       << "\n"; // s
            file << "    unaccounted: "         << unaccounted()                     << "\n"; // s
            file.close();
        }
    }

    // Record statistics based on the 'bhir'
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
    std::string fuseCacheHits() {
        return pprint_ratio(fuser_cache_lookups - fuser_cache_misses, fuser_cache_lookups);
    }

    std::string codegenCacheHits() {
        return pprint_ratio(codegen_cache_lookups - codegen_cache_misses, codegen_cache_lookups);
    }

    std::string kernelCacheHits() {
        return pprint_ratio(kernel_cache_lookups - kernel_cache_misses, kernel_cache_lookups);
    }

    std::string arrayContractions() {
        return pprint_ratio(num_temp_arrays, num_base_arrays);
    }

    std::string outerFusionRatio() {
        return pprint_ratio(num_blocks_out_of_fuser, num_instrs_into_fuser);
    }

    std::string MallocCacheHits() {
        return pprint_ratio(malloc_cache_lookups - malloc_cache_misses, malloc_cache_lookups);
    }

    double memoryUsage() {
        return max_memory_usage / 1024 / 1024;
    }

    double throughput() {
        return (double) totalwork / (double) wallclock.count();
    }

    double workBelowThredshold() {
        return (double) threading_below_threshold / (double) totalwork * 100.0;
    }

    double timeOther() {
        return (time_total_execution - time_pre_fusion - time_fusion - time_codegen - time_compile - time_exec
                - time_copy2dev - time_copy2host - time_offload).count();
    }

    double unaccounted() {
        return (wallclock - time_total_execution).count();
    }
};

} // jitk
} // bohrium
