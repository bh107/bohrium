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
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

#include <bh.h>
#include <bh_serialize.h>
#include "comm.h"

#include "zlib.h"


using boost::asio::ip::tcp;
using namespace std;
using namespace bohrium;

namespace bohrium {
namespace proxy {

boost::asio::io_service io_service;
tcp::socket socket(io_service);

static void init_client_socket(tcp::socket &socket, const std::string &address, int port)
{
    const unsigned int retries = 100;
    for(unsigned int i = 1; i <= retries; ++i)
    {
        try
        {
            cout << "[PROXY-VEM] Connecting to " << address << ":" << port << endl;
            // Get a list of endpoints corresponding to the server name.
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(address, to_string(port));
            tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);
            tcp::resolver::iterator end;

            // Try each endpoint until we successfully establish a connection.
            boost::system::error_code error = boost::asio::error::host_not_found;
            while (error && endpoint_iterator != end)
            {
                socket.close();
                socket.connect(*endpoint_iterator++, error);
            }
            if (error)
                throw boost::system::system_error(error);
            socket.set_option(boost::asio::ip::tcp::no_delay(true));
            return;
        }
        catch(...)
        {
            this_thread::sleep_for(chrono::seconds(1));
            cout << "Retrying - attempt number " << i << " of " << retries << endl;
        }
    }
}

static void init_server_socket(tcp::socket &socket, int port)
{
    cout << "[PROXY-VEM] Server listen on port " << port << endl;
    tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), port));
    acceptor.accept(socket);
    socket.set_option(boost::asio::ip::tcp::no_delay(true));
}

static void comm_send_array_data(const bh_base *base)
{
    assert(base->data != NULL);

    const size_t org_size = bh_base_size(base);
    size_t new_size = compressBound(org_size);
    vector<Bytef> buffer(new_size);
    compress(&buffer[0], &new_size, (Bytef*)base->data, org_size);

    const size_t size[] = {new_size};
    boost::asio::write(socket, boost::asio::buffer(size));
    boost::asio::write(socket, boost::asio::buffer(&buffer[0], new_size));
}

static void comm_recv_array_data(bh_base *base)
{
    assert(base->data != NULL);

    size_t size[1];
    boost::asio::read(socket, boost::asio::buffer(size));

    const size_t org_size = bh_base_size(base);
    size_t new_size = org_size;

    vector<char> compressed(size[0]);
    boost::asio::read(socket, boost::asio::buffer(compressed));

    uncompress((Bytef*)base->data, &new_size, (Bytef*) (&compressed[0]), compressed.size());
    assert(new_size == org_size);
}
/*
static void comm_send_array_dataOLD(const bh_base *base)
{
    cout << "comm_send_array_data()" << endl;
    assert(base->data != NULL);

    vector<char> compressed;
    {
        boost::iostreams::filtering_ostream os;
        os.push(boost::iostreams::gzip_compressor());
        os.push(boost::iostreams::back_inserter(compressed));

        char *data = reinterpret_cast<char*>(const_cast<bh_base*>(base)->data);
        boost::iostreams::basic_array_source<char> source(data, bh_base_size(base));
        boost::iostreams::stream<boost::iostreams::basic_array_source <char> > input_stream(source);
        os << input_stream;
        boost::iostreams::flush(os);
        boost::iostreams::close(os);
    }
    cout << "decompressed size: " << bh_base_size(base) << endl;
    const size_t size[] = {compressed.size()};
    cout << "sending compressed size: " << size[0] << endl;
    boost::asio::write(socket, boost::asio::buffer(size));
    cout << "sending compressed data " << endl;
    boost::asio::write(socket, boost::asio::buffer(compressed, compressed.size()));
    cout << "comm_send_array_data() END" << endl;
}

static void comm_recv_array_dataOLD(bh_base *base)
{
    cout << "comm_recv_array_data()" << endl;
    assert(base->data != NULL);

    size_t size[1];
    boost::asio::read(socket, boost::asio::buffer(size));
    cout << "receiving compressed size: " << size[0] << endl;

    vector<char> compressed(size[0]);
    boost::asio::read(socket, boost::asio::buffer(compressed));
    cout << "received data, let's decompress!" << endl;

    vector<char> decompressed;
    {
        boost::iostreams::filtering_ostream os;
        os.push(boost::iostreams::gzip_decompressor());
        os.push(boost::iostreams::back_inserter(decompressed));

        //boost::iostreams::write(os, &compressed[0], compressed.size());

        boost::iostreams::basic_array_source<char> source(&compressed[0], compressed.size());
        boost::iostreams::stream<boost::iostreams::basic_array_source <char> > input_stream(source);
        os << input_stream;
        boost::iostreams::flush(os);
        boost::iostreams::close(os);
    }
    cout << "decompressed size: " << decompressed.size() << " == array size: " << bh_base_size(base) << endl;
    assert(decompressed.size() == (size_t)bh_base_size(base));
    std::memcpy(base->data, &decompressed[0], decompressed.size());


    cout << "comm_recv_array_data() END" << endl;
}
*/

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
    boost::asio::write(socket, boost::asio::buffer(buf_head));
    boost::asio::write(socket, boost::asio::buffer(buf_body));

    //Send array data
    for(size_t i=0; i< data_send.size(); ++i)
    {
        bh_base *base = data_send[i];
        assert(base->data != NULL);
        send_array_data(base);
    }

    //Cleanup discard base array etc.
    exec_serializer.cleanup(bhir);

    //Receive sync'ed array data
    for(size_t i=0; i< data_recv.size(); ++i)
    {
        bh_base *base = data_recv[i];
        bh_data_malloc(base);
        recv_array_data(base);
    }
}

void CommFrontend::send_array_data(const bh_base *base)
{
    assert(base->data != NULL);
    //boost::asio::write(socket, boost::asio::buffer(base->data, bh_base_size(base)));
    comm_send_array_data(base);
}

void CommFrontend::recv_array_data(bh_base *base)
{
    assert(base->data != NULL);
    //boost::asio::read(socket, boost::asio::buffer(base->data, bh_base_size(base)));
    comm_recv_array_data(base);
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
    comm_send_array_data(base);
}

void CommBackend::recv_array_data(bh_base *base)
{
    assert(base->data != NULL);
    comm_recv_array_data(base);
}

}}
