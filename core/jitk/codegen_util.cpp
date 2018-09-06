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

#include <limits>
#include <iomanip>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <boost/filesystem/operations.hpp>
#include <jitk/codegen_util.hpp>
#include <jitk/view.hpp>
#include <jitk/instruction.hpp>

using namespace std;

namespace bohrium {
namespace jitk {

string hash_filename(uint64_t compilation_hash, size_t source_hash, string extension) {
    stringstream ss;
    ss << setfill ('0') << setw(sizeof(size_t)*2) << hex << compilation_hash << "_"<< source_hash << extension;
    return ss.str();
}

boost::filesystem::path write_source2file(const std::string &src,
                                          const boost::filesystem::path &dir,
                                          const std::string &filename,
                                          bool verbose) {
    boost::filesystem::path srcfile = dir;
    srcfile /= filename;
    ofstream ofs(srcfile.string());
    ofs << src;
    ofs.flush();
    ofs.close();
    if (verbose) {
        cout << "Write source " << srcfile << endl;
    }
    return srcfile;
}

boost::filesystem::path get_tmp_path(const ConfigParser &config) {
    boost::filesystem::path tmp_path, unique_path;
    const boost::filesystem::path tmp_dir = config.defaultGet<boost::filesystem::path>("tmp_dir", "");
    if (tmp_dir.empty()) {
        tmp_path = boost::filesystem::temp_directory_path();
    } else {
        tmp_path = boost::filesystem::path(tmp_dir);
    }
    // On some systems `boost::filesystem::unique_path()` throws a runtime error
    // when `LC_ALL` is undefined (or invalid). In this case, we set `LC_ALL=C` and try again.
    try {
        unique_path = boost::filesystem::unique_path("bh_%%%%");
    } catch(std::runtime_error &e) {
        setenv("LC_ALL", "C", 1); // Force C locale
        unique_path = boost::filesystem::unique_path("bh_%%%%");
    }
    return tmp_path / unique_path;
}

void create_directories(const boost::filesystem::path &path) {
    constexpr int tries = 5;
    for (int i = 1; i <= tries; ++i) {
        try {
            boost::filesystem::create_directories(path);
            return;
        } catch (boost::filesystem::filesystem_error &e) {
            if (i == tries) {
                throw;
            }
            this_thread::sleep_for(chrono::seconds(3));
            cerr << e.what() << endl;
            cout << "Warning: " << e.what() << " (" << i << " attempt)" << endl;
        }
    }
}

std::vector<InstrPtr> order_sweep_set(const std::set<InstrPtr> &sweep_set, const SymbolTable &symbols) {
    vector<InstrPtr> ret;
    ret.reserve(sweep_set.size());
    std::copy(sweep_set.begin(),  sweep_set.end(), std::back_inserter(ret));
    std::sort(ret.begin(), ret.end(),
             [symbols](const InstrPtr & a, const InstrPtr & b) -> bool
             {
                 return symbols.viewID(a->operand[0]) > symbols.viewID(b->operand[0]);
             });
    return ret;
}

bool row_major_access(const bh_view &view) {
    if(not bh_is_constant(&view)) {
        assert(view.ndim > 0);
        for(int64_t i = 1; i < view.ndim; ++ i) {
            if (view.stride[i] > view.stride[i-1]) {
                return false;
            }
        }
    }
    return true;
}

bool row_major_access(const bh_instruction &instr) {
    for (const bh_view &view: instr.operand) {
        if (not row_major_access(view)) {
            return false;
        }
    }
    return true;
}

void to_column_major(std::vector<bh_instruction> &instr_list) {
    for(bh_instruction &instr: instr_list) {
        if (instr.opcode < BH_MAX_OPCODE_ID and row_major_access(instr)) {
            instr.transpose();
        }
    }
}

} // jitk
} // bohrium
