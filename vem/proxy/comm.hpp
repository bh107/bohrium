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
next_message_body(void *buffer, size_t size)
If not, see <http://www.gnu.org/licenses/>.
*/

#include <string>
#include <boost/asio.hpp>

#include <bh_serialize.hpp>

#ifndef __BH_VEM_PROXY_COMM_H
#define __BH_VEM_PROXY_COMM_H

class CommFrontend
{
private:
    bohrium::serialize::ExecuteFrontend exec_serializer;

    boost::asio::io_service io_service;
    boost::asio::ip::tcp::socket socket;
public:
    CommFrontend(int stack_level, const std::string &address, int port);
    ~CommFrontend();
    void execute(bh_ir &bhir);
    void send_array_data(const bh_base *base);
    void recv_array_data(bh_base *base);
};

class CommBackend
{
private:
    boost::asio::io_service io_service;
    boost::asio::ip::tcp::socket socket;
public:
    CommBackend();
    ~CommBackend();
    CommBackend(const std::string &address, int port=4200);
    bohrium::serialize::Header next_message_head();
    void next_message_body(std::vector<char> &buffer);
    void send_array_data(const bh_base *base);
    void recv_array_data(bh_base *base);
};

#endif
