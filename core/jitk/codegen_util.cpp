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

pair<uint32_t, uint32_t> work_ranges(uint64_t work_group_size, int64_t block_size) {
    if (numeric_limits<uint32_t>::max() <= work_group_size or
        numeric_limits<uint32_t>::max() <= block_size or
        block_size < 0) {
        throw runtime_error("work_ranges(): sizes cannot fit in a uint32_t!");
    }
    const auto lsize = (uint32_t) work_group_size;
    const auto rem   = (uint32_t) block_size % lsize;
    const auto gsize = (uint32_t) block_size + (rem==0?0:(lsize-rem));
    return make_pair(gsize, lsize);
}

} // jitk
} // bohrium
