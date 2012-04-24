/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __STATICSTORE_HPP
#define __STATICSTORE_HPP

#include <cstdlib>
#include <stdexcept>
#include <deque>

#ifdef DEBUG
#include <iostream>
#include <typeinfo>
#endif


#define __SC_DEFAULT_BUFFER_SIZE (4096)

class StaticStoreException : public std::exception {};


template <typename T>
class StaticStore
{
private:
    T* buffer;
    T* nextElement;
    size_t bufferSize;
    std::deque<T*> emptySlot;
    int counter;
  public:
    StaticStore(size_t size);
    StaticStore();
    ~StaticStore();
    void clear();
    void erase(T* e);
#ifndef _WIN32
    template <typename... As>
    T* next(As... as);
#endif
    T* c_next();
};


template <typename T>
StaticStore<T>::StaticStore(size_t initialSize) :
    bufferSize(initialSize)
{
    buffer = (T*)malloc(initialSize*sizeof(T));
    if (buffer == NULL)
    {
        throw std::runtime_error("Out of memory");
    }
    nextElement = buffer;
    counter = 0;
#ifdef DEBUG
    std::cout << "StaticStore<" << typeid(T).name() << ">(): ";
    std::cout << "\n  buffer: " << buffer;
    std::cout << "\n  bufferSize: " << bufferSize;
    std::cout << "\n  dataSize: " << bufferSize*sizeof(T) << std::endl;
#endif
}

template <typename T>
StaticStore<T>::StaticStore()
{
    StaticStore(__SC_DEFAULT_BUFFER_SIZE);
}

template <typename T>
StaticStore<T>::~StaticStore()
{
    if(counter > 0)
        std::cout << "[VEM-NODE] Warning " << counter << " arrays was not destroyed by the BRIDGE" << std::endl;
    free(buffer);
}

template <typename T>
T* StaticStore<T>::c_next()
{
    counter++;
    if (!emptySlot.empty())
    {
        T* ref = emptySlot.front();
        emptySlot.pop_front();
        return ref;
    }
    else if (nextElement < buffer + bufferSize)
    {
        return nextElement++;
    }
    else
    {
        throw StaticStoreException();
    }
}

#ifndef _WIN32
template <typename T>
template <typename... As>
T* StaticStore<T>::next(As... as)
{
    counter++;
    if (!emptySlot.empty())
    {
        T* ref = emptySlot.front();
        emptySlot.pop_front();
        return new(ref) T(as...);
    }
    else if (nextElement < buffer + bufferSize)
    {
        return new(nextElement++) T(as...);
    }
    else
    {
        throw StaticStoreException();
    }
}
#endif

template <typename T>
void StaticStore<T>::clear()
{
    counter = 0;
    nextElement = buffer;
    emptySlot.clear();
}

template <typename T>
void StaticStore<T>::erase(T* e)
{
    counter--;
    emptySlot.push_back(e);
}

#endif
