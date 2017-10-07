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

#include <vector>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/set.hpp>
#include <bh_view.hpp>
#include <bh_ir.hpp>
#include <bh_instruction.hpp>

// Forward declaration of class boost::serialization::access
namespace boost {namespace serialization {class access;}}

namespace msg {

/* Message type */
enum class Type
{
    INIT,
    SHUTDOWN,
    EXEC,
    GET_DATA,
    MSG,
    EXTMETHOD
};

struct Header
{
    Type type;
    size_t body_size;
    Header(Type type, size_t body_size):type(type),body_size(body_size){}
    Header(const std::vector<char> &buffer);
    void serialize(std::vector<char> &buffer);
};
constexpr size_t HeaderSize = sizeof(Type) + sizeof(size_t);

struct Init
{
    int stack_level;// Stack level of the component
    Init(int stack_level):stack_level(stack_level){}
    Init(const std::vector<char> &buffer);

    void serialize(std::vector<char> &buffer);
};

struct GetData
{
    bh_base *base;
    bool nullify;
    GetData(bh_base *base, bool nullify):base(base), nullify(nullify) {}
    GetData(const std::vector<char> &buffer);

    void serialize(std::vector<char> &buffer);
};

}