/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
http://bohrium.bitbucket.org

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
#ifndef __BOHRIUM_BRIDGE_CPP_MAPPING
#define __BOHRIUM_BRIDGE_CPP_MAPPING

namespace bxx {

template <typename TO, typename TL, typename TR>
multi_array<TO>& gather(multi_array<TO>& out,
                        multi_array<TL>& in, multi_array<TR>& index)
{
    // TODO: Verify that bases are distinct.
    if (!((out.getRank() == 1) && (in.getRank() == 1) && (index.getRank() == 1))) {
        throw std::runtime_error("Unsupported rank.");
    }
    if (out.len() != index.len()) {
        throw std::runtime_error("Unsupported shape.");
    }

    return bh_gather(out, in, index);
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& scatter(multi_array<TO>& out,
                        multi_array<TL>& in, multi_array<TR>& index)
{
    // TODO: Verify that bases are distinct.
    if (!((out.getRank() == 1) && (in.getRank() == 1) && (index.getRank() == 1))) {
        throw std::runtime_error("Unsupported rank.");
    }
    if (in.len() != index.len()) {
        throw std::runtime_error("Unsupported shape.");
    }

    return bh_scatter(out, in, index);
}

}
#endif

