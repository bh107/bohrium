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

#define BH_TIMING
#include <bh_timing.hpp>
#undef BH_TIMING

std::ostream& operator<< (std::ostream& os, bh::timing2 const& t)
{
    os << t.start << " " << t.end << std::endl;
    return os;
}

std::ostream& operator<< (std::ostream& os, bh::timing4 const& t)
{
    os << t.queued << " " << t.submit << " " << t.start << " " << t.end << std::endl;
    return os;
}
