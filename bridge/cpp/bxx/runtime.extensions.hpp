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
#ifndef __BOHRIUM_BRIDGE_CPP_EXTENSIONS
#define __BOHRIUM_BRIDGE_CPP_EXTENSIONS
namespace bxx {

void bh_ext_visualizer(multi_array<float> ary, multi_array<float> args)
{
    Runtime::enqueue_extension(
        "visualizer",
        ary,
        args,
        ary
    );
}

template <typename T>
multi_array<T>& matmul(multi_array<T>& res, multi_array<T>& lhs, multi_array<T>& rhs) {
    if (!(same_shape(res, lhs) && (same_shape(lhs, rhs)))) {
        throw std::runtime_error("Incompatible shapes of output and input.");
    }

    Runtime::enqueue_extension("matmul", res, lhs, rhs);

    return res;
}

}
#endif

