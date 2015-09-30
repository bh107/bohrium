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

#include <iostream>
#include <boost/asio.hpp>

#include <bh.h>
#include <bh_serialize.h>
#include "comm.h"

using boost::asio::ip::tcp;
using namespace std;
using namespace bohrium;

namespace bohrium {
namespace proxy {

boost::asio::io_service io_service;
tcp::socket socket(io_service);

static void init_client_socket(tcp::socket &socket, const std::string &address, int port)
{
    for(unsigned int i = 0; i < 1000000; ++i)
    {
        try
        {
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(address, to_string(port));
            tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
            boost::asio::connect(socket, endpoint_iterator);
            break;
        }
        catch(...)
        {
            cout << "retry"  << endl;

        }

    }
}

static void init_server_socket(tcp::socket &socket, int port)
{
    tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), port));
    acceptor.accept(socket);
}

CommFrontend::CommFrontend(const char* component_name, const std::string &address, int port)
{
    init_client_socket(socket, address, port);

    //Serialize message body
    vector<char> buf_body;
    serialize::Init body(component_name);
    body.serialize(buf_body);

    //Serialize message head
    vector<char> buf_head;
    serialize::Header head(serialize::TYPE_INIT, buf_body.size());
    head.serialize(buf_head);

    //Send serialized message
    cout << "server send INIT message " << endl;
    boost::asio::write(socket, boost::asio::buffer(buf_head));
    boost::asio::write(socket, boost::asio::buffer(buf_body));
}

void CommFrontend::shutdown()
{
    //Serialize message head
    vector<char> buf_head;
    serialize::Header head(serialize::TYPE_SHUTDOWN, 0);
    head.serialize(buf_head);

    //Send serialized message
    cout << "server send SHUTDOWN message " << endl;
    boost::asio::write(socket, boost::asio::buffer(buf_head));
    socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
    socket.close();

}

void CommFrontend::execute(bh_ir &bhir)
{
    //Serialize the BhIR
    vector<char> buf_body;
    vector<bh_base*> data_send;
    vector<bh_base*> data_recv;
    exec_serializer.serialize(bhir, buf_body, data_send, data_recv);

    //Serialize message head
    vector<char> buf_head;
    serialize::Header head(serialize::TYPE_EXEC, buf_body.size());
    head.serialize(buf_head);

    //Send serialized message
    cout << "server send EXEC message " << endl;
    boost::asio::write(socket, boost::asio::buffer(buf_head));
    boost::asio::write(socket, boost::asio::buffer(buf_body));

    //Send array data
    cout << "EXEC send new base data: ";
    for(size_t i=0; i< data_send.size(); ++i)
    {
        bh_base *base = data_send[i];
        assert(base->data != NULL);
        printf("%p ", base);
        send_array_data(base);
    }
    cout << endl;

    //Cleanup discard base array etc.
    exec_serializer.cleanup(bhir);

    //Receive sync'ed array data
    cout << "EXEC recv sync'ed base data: [";
    for(size_t i=0; i< data_recv.size(); ++i)
    {
        bh_base *base = data_recv[i];
        bh_data_malloc(base);
        printf("%p ", base);
        recv_array_data(base);
    }
    cout << "]" << endl;
}

void CommFrontend::send_array_data(const bh_base *base)
{
    assert(base->data != NULL);
    boost::asio::write(socket, boost::asio::buffer(base->data, bh_base_size(base)));
}

void CommFrontend::recv_array_data(bh_base *base)
{
    assert(base->data != NULL);
    boost::asio::read(socket, boost::asio::buffer(base->data, bh_base_size(base)));
}


CommBackend::CommBackend(const std::string &address, int port) {
    init_server_socket(socket, port);
}

serialize::Header CommBackend::next_message_head()
{
    //Let's read the head of the message
    vector<char> buf_head(serialize::HeaderSize);
    boost::asio::read(socket, boost::asio::buffer(buf_head));
    serialize::Header head(buf_head);

    cout << "client read head with body size of " << head.body_size << endl;
    return head;
}

void CommBackend::next_message_body(std::vector<char> &buffer)
{
    boost::asio::read(socket, boost::asio::buffer(buffer));
}

void CommBackend::shutdown()
{
    socket.shutdown(boost::asio::ip::tcp::socket::shutdown_both);
    socket.close();
}

void CommBackend::send_array_data(const bh_base *base)
{
    assert(base->data != NULL);
    boost::asio::write(socket, boost::asio::buffer(base->data, bh_base_size(base)));
}

void CommBackend::recv_array_data(bh_base *base)
{
    assert(base->data != NULL);
    boost::asio::read(socket, boost::asio::buffer(base->data, bh_base_size(base)));
}

}}
