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
#pragma once

#include <string>
#include <boost/asio.hpp>

#include "serialize.hpp"

class CommFrontend {
    uint64_t sim_bandwidth = 1000; // bytes per second
public:
    boost::asio::io_service io_service;
    boost::asio::ip::tcp::socket socket;

    CommFrontend(int stack_level, const std::string &address, int port, uint64_t sim_bandwidth);

    ~CommFrontend();

    /// Write to the `CommBackend`
    void write(const std::vector<char> &buf) {
        boost::asio::write(socket, boost::asio::buffer(buf));
    }

    /// Read string from the `CommBackend`
    std::string read();

    /// Send data to the `CommBackend`
    void send_data(const std::vector<unsigned char> &data);

    /// Receive data from the `CommBackend`
    std::vector<unsigned char> recv_data();

    std::string hostname() const {
        return boost::asio::ip::host_name();
    }

    std::string ip() const {
        std::stringstream ss;
        ss << socket.local_endpoint().address() << "\n";
        return ss.str();
    }
};

class CommBackend {
private:
    boost::asio::io_service io_service;
    boost::asio::ip::tcp::socket socket;
public:
    ~CommBackend();

    CommBackend(const std::string &address, int port = 4200);

    /// Read from the `CommFrontend`
    void read(std::vector<char> &buf) {
        boost::asio::read(socket, boost::asio::buffer(buf));
    }

    /// Write string to the `CommFrontend`
    void write(const std::string &str) {
        // Write the whole string including the `\0` terminator
        boost::asio::write(socket, boost::asio::buffer(str.c_str(), str.size() + 1));
    }

    /// Send data to the `CommFrontend`
    void send_data(const std::vector<unsigned char> &data);

    /// Receive data from the `CommFrontend`
    std::vector<unsigned char> recv_data();

    std::string hostname() const {
        return boost::asio::ip::host_name();
    }

    std::string ip() const {
        std::stringstream ss;
        ss << socket.local_endpoint().address() << "\n";
        return ss.str();
    }
};
