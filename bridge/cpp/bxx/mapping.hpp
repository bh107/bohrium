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
multi_array<TO>& gatherz(multi_array<TO>& out,
                         multi_array<TL>& in, multi_array<TR>& index)
{
    if (!((out.getRank() == 1) && (in.getRank() == 1) && (index.getRank() == 1))) {
        throw std::runtime_error("Unsupported arguments: One of [out, in, index] rank != 1.");
    }
    if (out.getBase() == in.getBase()) {
        throw std::runtime_error("Unsupported arguments: 'out' shares base/data with 'in'.");
    }
    if (index.len() > out.len()) {
        throw std::runtime_error("Unsupported index: index.len() > out.len()");
    }
    
    if (index.len() == out.len()) {
        bh_gather(out, in, index);
        return out;
    } else {
        bh_gather(out[_(0, index.len()-1)], in, index); 
        return out; 
    }
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& gather(multi_array<TO>& out,
                        multi_array<TL>& in, multi_array<TR>& index)
{
    if (index.len() != out.len()) {
        throw std::runtime_error(
            "Unsupported arguments: index.len() != in.len() "
            " perhaps you want gatherz(out, in, index) instead?");
    }

    return gatherz(out, in, index);
}

template <typename TO, typename TL, typename TR>
multi_array<TO>& scatterz(multi_array<TO>& out,
                          multi_array<TL>& in, multi_array<TR>& index)
{
    if (!((out.getRank() == 1) && (in.getRank() == 1) && (index.getRank() == 1))) {
        throw std::runtime_error("Unsupported arguments: One of [out, in, index] rank != 1.");
    }
    if (out.getBase() == in.getBase()) {
        throw std::runtime_error("Unsupported arguments: 'out' shares base/data with 'in'.");
    }
    if (index.len() > in.len()) {
        throw std::runtime_error("Unsupported index: index.len() > in.len()");
    }

    if (index.len() == in.len()) {
        bh_scatter(out, in, index);
        return out;
    } else {
        bh_scatter(out, in[_(0, index.len()-1)], index);
        return out;
    }

}

template <typename TO, typename TL, typename TR>
multi_array<TO>& scatter(multi_array<TO>& out,
                        multi_array<TL>& in, multi_array<TR>& index)
{
    if (index.len() != in.len()) {
        throw std::runtime_error(
            "Unsupported arguments: index.len() != in.len() "
            " perhaps you want scatterz(out, in, index) instead?");
    }

    return scatterz(out, in, index);
}

}
#endif

