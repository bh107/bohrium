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

#include <algorithm>
#include <vector>
#include <type_traits>

#include <bh_util.hpp>

#include <boost/iterator/filter_iterator.hpp>

using namespace std;
namespace fs = boost::filesystem;

namespace util {

void remove_old_files(const fs::path &dir, int64_t num_of_newest_to_keep) {
    fs::directory_iterator dir_first(dir), dir_last;
    std::vector<fs::path> files;

    auto pred = [](const fs::directory_entry& p)
    {
        return fs::is_regular_file(p);
    };

    std::copy(boost::make_filter_iterator(pred, dir_first, dir_last),
              boost::make_filter_iterator(pred, dir_last, dir_last),
              std::back_inserter(files)
    );

    std::sort(files.begin(), files.end(),
              [](const fs::path& p1, const fs::path& p2)
              {
                  return fs::last_write_time(p1) < fs::last_write_time(p2);
              });

    for(int64_t i=num_of_newest_to_keep; i<static_cast<int64_t>(files.size()); ++i) {
        fs::remove(files[i]);
    }
}

} // util
