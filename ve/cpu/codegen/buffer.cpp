#include <sstream>
#include "utils.hpp"
#include "codegen.hpp"

using namespace std;

namespace kp{
namespace engine{
namespace codegen{

Buffer::Buffer(void) : buffer_(NULL), id_(0) {}
Buffer::Buffer(kp_buffer* buffer, int64_t buffer_id) : buffer_(buffer), id_(buffer_id) {
    if (NULL == buffer_) {
        throw runtime_error("Constructing a NULL Buffer, when expecting to have one");
    }
}

string Buffer::name(void)
{
    stringstream ss;
    ss << "buf" << id_;
    return ss.str();
}

string Buffer::data(void)
{
    stringstream ss;
    ss << name() << "_data";
    return ss.str();
}

string Buffer::nelem(void)
{
    stringstream ss;
    ss << name() << "_nelem";
    return ss.str();
}

string Buffer::etype(void)
{
    return kp::core::etype_to_ctype_text(kp::core::bhtype_to_etype(meta().type));
}

kp_buffer& Buffer::meta(void)
{
    return *buffer_;
}

int64_t Buffer::id(void)
{
    return id_;
}

}}}

