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
#ifndef __BOHRIUM_BRIDGE_CPP_MOVEMENT
#define __BOHRIUM_BRIDGE_CPP_MOVEMENT

namespace bxx {

template <typename T>
T* bh_data_export(multi_array<T>& op, Export::Option option)
{
    bh_sync(op);                                            // Ensure operations..
    Runtime::instance().flush();                            // .. are completed.

    if (!op.initialized()) {
        throw std::runtime_error("Array is non-initialized => no relation to data.");
    }

    bh_base* base = op.getBase();
    T* data = (T*)base->data;                               // Grab data-pointer

    //
    // Handle export options
    //
    
    if ((NULL==data) && ((option & Export::WO_ALLOC)==0)) { // Allocate data
        bh_error res = bh_data_malloc(base);
        data = (T*)base->data;
        if ((BH_SUCCESS!=res) || (NULL==data)) {             // Verify allocation
            throw std::runtime_error("Export error: allocation failed.");
        }
        if ((option & Export::WO_ZEROING)==0) {             // Zero-initialize by..
            bh_identity(op, (T)0);                          // ..default, to ensure..
            bh_sync(op);                                    // ..first-touch policy.
            bh_discard(op);                                     
            Runtime::instance().flush();
        }
    }

    if ((data) && ((option & Export::RELEASED)>0)) {        // Release it from Bohrium
        base->data = NULL;
    }
    bh_free(op);

    // Note: The returned data-pointer might still be NULL at this point.
    //       If allocation is disabled.

    return data;
}

template <typename T>
void bh_data_import(multi_array<T>& op, T* data)
{
    if (NULL==data) {
        throw std::runtime_error(
            "Trying to import NULL. What are you trying to do here?");
    }

    if (!op.initialized()) {
        throw std::runtime_error(
            "multi_array is  non-initialized => Initialize before importing.");
    }
    
    bh_base* base = op.getBase();
    if (base->data == data) {       // All is good
        return;
    }
    
    //
    // Importing "foreign" data.
    //
    if(!is_aligned(data, 16) ) {                    // Ensure alignment
        std::cout << "Importing unaligned data, regressing to copy." << std::endl;
        size_t nbytes = (base->nelem) * bh_type_size(base->type);
        if (!op.allocated()) {                      // Allocate aligned
            base->data = bh_memory_malloc(nbytes);
        }
        memcpy(base->data, data, nbytes);           // Copy the data

        // TODO: This still messes up first-touch/numa.
    }
}

}
#endif

