/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
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
  public:
    StaticStore(size_t size);
    StaticStore();
    ~StaticStore();
    void clear();
    void erase(T* e);
    template <typename... As>
    T* next(As... as);
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
    free(buffer);
}

template <typename T> 
T* StaticStore<T>::c_next()
{
    if (nextElement) 
    {
        if (nextElement >= buffer + bufferSize)
        {
            nextElement = 0;
        }
        else
        {
            return nextElement++;
        }
    }
    if (emptySlot.empty())
    {
        throw StaticStoreException();
    }
    T* ref = emptySlot.front();
    emptySlot.pop_front();
    return ref;

}


template <typename T> 
template <typename... As>
T* StaticStore<T>::next(As... as)
{
    if (nextElement) 
    {
        if (nextElement >= buffer + bufferSize)
        {
            nextElement = 0;
        }
        else
        {
            return new(nextElement++) T(as...);
        }
    }
    if (emptySlot.empty())
    {
        throw StaticStoreException(0);
    }
    T* ref = emptySlot.front();
    emptySlot.pop_front();
    return new(ref) T(as...);
}

template <typename T> 
void StaticStore<T>::clear()
{
    nextElement = buffer;
    emptySlot.clear();
}

template <typename T> 
void StaticStore<T>::erase(T* e)
{
    emptySlot.push_back(e);
}

#endif
