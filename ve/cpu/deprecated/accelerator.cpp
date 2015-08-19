#include <cinttypes>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <omp.h>
#include "accelerator.hpp"

#ifdef DEBUGGING
#define DEBUG(tag,x) do { std::cerr << TAG << "::" << x << std::endl; } while (0);
#else
#define DEBUG(tag,x)
#endif

using namespace std;

namespace kp{
namespace engine{

const char Accelerator::TAG[] = "Accelerator";

Accelerator::Accelerator(int id) : id_(id), bytes_allocated_(0) {}

int Accelerator::id(void)
{
    return id_;
}

string Accelerator::text(void)
{
    stringstream ss;
    ss << "Accelerator id(" << id_ << ") {}";
    return ss.str();
}

size_t Accelerator::bytes_allocated(void)
{
    return bytes_allocated_;
}

bool Accelerator::allocated(kp_operand & operand)
{
    return buffers_allocated_.count(operand.base)!=0;
}

bool Accelerator::pushed(kp_operand & operand)
{
    return buffers_pushed_.count(operand.base)!=0;
}

#if defined(VE_CPU_WITH_INTEL_LEO)
bool Accelerator::offloadable(void)
{
    // TODO: Actual checks, such as whether there are devices available.
    return true;
}

template <typename T>
void Accelerator::_alloc(kp_operand& kp_operand)
{
    DEBUG(TAG, "_alloc bh_base(" << (void*)(kp_operand.base) << ") bh_base.data(" << kp_operand.base->data << ")");

    if (allocated(kp_operand)) {                   // Operand is already allocated
        DEBUG(TAG, "_alloc skipping");
        return;
    }

    T* data = (T*)(kp_operand.base->data);         // Grab the buffer
    const int nelem = kp_operand.nelem;

    bytes_allocated_ += nelem*sizeof(T);        // House-keeping
    buffers_allocated_.insert(kp_operand.base);

    #pragma offload_transfer                                    \
            target(mic:id_)                                     \
            nocopy(data:length(nelem) alloc_if(1) free_if(0))
}

void Accelerator::alloc(kp_operand& kp_operand)
{
    switch(kp_operand.etype) {
        case KP_BOOL:      _alloc<unsigned char>(kp_operand); break;
        case KP_INT8:      _alloc<int8_t>(kp_operand); break;
        case KP_INT16:     _alloc<int16_t>(kp_operand); break;
        case KP_INT32:     _alloc<int32_t>(kp_operand); break;
        case KP_INT64:     _alloc<int64_t>(kp_operand); break;
        case KP_UINT8:     _alloc<uint8_t>(kp_operand); break;
        case KP_UINT16:    _alloc<uint16_t>(kp_operand); break;
        case KP_UINT32:    _alloc<uint32_t>(kp_operand); break;
        case KP_UINT64:    _alloc<uint64_t>(kp_operand); break;
        case KP_FLOAT32:   _alloc<float>(kp_operand); break;
        case KP_FLOAT64:   _alloc<double>(kp_operand); break;

        case KP_COMPLEX64:
        case KP_COMPLEX128:
        case KP_PAIRLL:
        default:
            throw runtime_error("Accelerator does not support this etype, yet...");
            break;
    }
}

template <typename T>
void Accelerator::_free(kp_operand& kp_operand)
{
    DEBUG(TAG, "_free bh_base(" << (void*)(kp_operand.base) << ") bh_base.data(" << kp_operand.base->data << ")");

    if (!allocated(kp_operand)) {              // Not allocated on device
        DEBUG(TAG, "_free skipping...");
        return;
    }

    T* data = (T*)(kp_operand.base->data);
    const int nelem = kp_operand.nelem;

    bytes_allocated_ -= nelem*sizeof(T);    // House-keeping
    buffers_allocated_.erase(kp_operand.base);
    buffers_pushed_.erase(kp_operand.base);

    #pragma offload_transfer                                    \
            target(mic:id_)                                     \
            nocopy(data:length(nelem) alloc_if(0) free_if(1))
}

void Accelerator::free(kp_operand& kp_operand)
{
    switch(kp_operand.etype) {
        case KP_BOOL:      _free<unsigned char>(kp_operand); break;
        case KP_INT8:      _free<int8_t>(kp_operand); break;
        case KP_INT16:     _free<int16_t>(kp_operand); break;
        case KP_INT32:     _free<int32_t>(kp_operand); break;
        case KP_INT64:     _free<int64_t>(kp_operand); break;
        case KP_UINT8:     _free<uint8_t>(kp_operand); break;
        case KP_UINT16:    _free<uint16_t>(kp_operand); break;
        case KP_UINT32:    _free<uint32_t>(kp_operand); break;
        case KP_UINT64:    _free<uint64_t>(kp_operand); break;
        case KP_FLOAT32:   _free<float>(kp_operand); break;
        case KP_FLOAT64:   _free<double>(kp_operand); break;

        case KP_COMPLEX64:
        case KP_COMPLEX128:
        case KP_PAIRLL:
        default:
            throw runtime_error("Accelerator does not support this etype, yet...");
            break;
    }
}

template <typename T>
void Accelerator::_push(kp_operand& kp_operand)
{
    DEBUG(TAG, "_push bh_base(" << (void*)(kp_operand.base) << ") bh_base.data(" << kp_operand.base->data << ")");
    if (pushed(kp_operand)) {                  // Already pushed to device
        DEBUG(TAG, "_push skipping..");
    }
    if (!allocated(kp_operand)) {
        // TODO: THROW UP
    }
    T* data = (T*)(kp_operand.base->data);
    const int nelem = kp_operand.nelem;

    buffers_pushed_.insert(kp_operand.base);   // House-keeping

    #pragma offload_transfer                                    \
            target(mic:id_)                                     \
            in(data:length(nelem) alloc_if(0) free_if(0))
}

void Accelerator::push(kp_operand& kp_operand)
{
    switch(kp_operand.etype) {
        case KP_BOOL:      _push<unsigned char>(kp_operand); break;
        case KP_INT8:      _push<int8_t>(kp_operand); break;
        case KP_INT16:     _push<int16_t>(kp_operand); break;
        case KP_INT32:     _push<int32_t>(kp_operand); break;
        case KP_INT64:     _push<int64_t>(kp_operand); break;
        case KP_UINT8:     _push<uint8_t>(kp_operand); break;
        case KP_UINT16:    _push<uint16_t>(kp_operand); break;
        case KP_UINT32:    _push<uint32_t>(kp_operand); break;
        case KP_UINT64:    _push<uint64_t>(kp_operand); break;
        case KP_FLOAT32:   _push<float>(kp_operand); break;
        case KP_FLOAT64:   _push<double>(kp_operand); break;

        case KP_COMPLEX64:
        case KP_COMPLEX128:
        case KP_PAIRLL:
        default:
            throw runtime_error("Accelerator does not support this etype, yet...");
            break;
    }
}

template <typename T>
void Accelerator::_pull(kp_operand& kp_operand)
{
    DEBUG(TAG, "_pull bh_base(" << (void*)(kp_operand.base) << ") bh_base.data(" << kp_operand.base->data << ")");
    if (buffers_allocated_.count(kp_operand.base)==0) {    // Not allocated on device
        DEBUG(TAG, "_pull skipping");
        return;
    }

    T* data = (T*)(kp_operand.base->data);
    const int nelem = kp_operand.nelem;

    #pragma offload_transfer                                    \
            target(mic:id_)                                     \
            out(data:length(nelem) alloc_if(0) free_if(0))
}

void Accelerator::pull(kp_operand& kp_operand)
{
    switch(kp_operand.etype) {
        case KP_BOOL:      _pull<unsigned char>(kp_operand); break;
        case KP_INT8:      _pull<int8_t>(kp_operand); break;
        case KP_INT16:     _pull<int16_t>(kp_operand); break;
        case KP_INT32:     _pull<int32_t>(kp_operand); break;
        case KP_INT64:     _pull<int64_t>(kp_operand); break;
        case KP_UINT8:     _pull<uint8_t>(kp_operand); break;
        case KP_UINT16:    _pull<uint16_t>(kp_operand); break;
        case KP_UINT32:    _pull<uint32_t>(kp_operand); break;
        case KP_UINT64:    _pull<uint64_t>(kp_operand); break;
        case KP_FLOAT32:   _pull<float>(kp_operand); break;
        case KP_FLOAT64:   _pull<double>(kp_operand); break;

        case KP_COMPLEX64:
        case KP_COMPLEX128:
        case KP_PAIRLL:
        default:
            throw runtime_error("Accelerator does not support this etype, yet...");
            break;
    }
}

int Accelerator::get_max_threads(void)
{
    int mthreads;                        
    #pragma offload target(mic:id_) out(mthreads)
    {                                    
        mthreads = omp_get_max_threads();
    }                                    
    return mthreads;                     
}

#else
//
// Fallback when compiled without Language Extensions for Offload (Intel LEO).
//
void Accelerator::alloc(kp_operand & operand) {}
void Accelerator::free(kp_operand & operand) {}
void Accelerator::push(kp_operand & operand) {}
void Accelerator::pull(kp_operand & operand) {}
int Accelerator::get_max_threads(void) { return 1; }
bool Accelerator::offloadable(void)
{
    return false;
}
#endif

}}

