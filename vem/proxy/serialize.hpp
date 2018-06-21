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
namespace boost { namespace serialization { class access; }}

namespace msg {

/** Message type */
enum class Type {
    INIT,
    SHUTDOWN,
    EXEC,
    GET_DATA,
    MEM_COPY,
    MSG
};

/** Message Header */
struct Header {
    Type type;
    size_t body_size;

    /** The regular constructor */
    Header(Type type, size_t body_size) : type(type), body_size(body_size) {}

    /** The de-serializing constructor */
    explicit Header(const std::vector<char> &buffer);

    /** Serialize to `buffer` */
    void serialize(std::vector<char> &buffer);
};

constexpr size_t HeaderSize = sizeof(Type) + sizeof(size_t);

/** RPC: the constructor (the first message send to initiate the backend) */
struct Init {
    int stack_level;// Stack level of the component

    /** The regular constructor */
    Init(int stack_level) : stack_level(stack_level) {}

    /** The de-serializing constructor */
    explicit Init(const std::vector<char> &buffer);

    /** Serialize to `buffer` */
    void serialize(std::vector<char> &buffer);
};

/** RPC: `getMemoryPointer()` */
struct GetData {
    bh_base *base;
    bool nullify;

    /** The regular constructor */
    GetData(bh_base *base, bool nullify) : base(base), nullify(nullify) {}

    /** The de-serializing constructor */
    explicit GetData(const std::vector<char> &buffer);

    /** Serialize to `buffer` */
    void serialize(std::vector<char> &buffer);
};

/** RPC: `memCopy()` */
struct MemCopy {
    bh_view src;
    std::string param;

    /** The regular constructor */
    explicit MemCopy(const bh_view &src, std::string param) : src(src), param(std::move(param)) {}

    /** The de-serializing constructor */
    explicit MemCopy(const std::vector<char> &buffer);

    /** Serialize to `buffer` */
    void serialize(std::vector<char> &buffer);
};

/** RPC: `message()` */
struct Message {
    std::string msg;

    /** The regular constructor */
    Message(std::string msg) : msg(std::move(msg)) {}

    /** The de-serializing constructor */
    explicit Message(const std::vector<char> &buffer);

    /** Serialize to `buffer` */
    void serialize(std::vector<char> &buffer);
};

}
