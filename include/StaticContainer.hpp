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

#ifndef __STATICCONTAINER_HPP
#define __STATICCONTAINER_HPP

#include <cstring>
#include <vector>
#include <stdexcept>
#include <cstdlib>

#define __SC_DEFAULT_BUFFER_SIZE (4096)

class StaticContainerException 
{
private:
    int code;
public:
    StaticContainerException(int code_) : code(code_) {}
};

template <typename T>
class StaticContainer 
{
private:
    T* buffer;
    T* nextElement;
    size_t bufferSize;
  public:
    typedef T* iterator;
    typedef T* reference;
    typedef const T* const_iterator;
    typedef const T* const_reference;
    StaticContainer(size_t size);
    StaticContainer();
    ~StaticContainer();
    virtual T* push_back(const T* e);
    virtual void pop_back();
    virtual void clear();
    virtual T* next();
    template <typename I>
    T* setNext(I i);
    virtual iterator begin();
    virtual iterator end();
    virtual iterator last();
    //virtual reference at(size_t n);
    virtual reference operator[] (size_t n);
};


template <typename T> 
StaticContainer<T>::StaticContainer(size_t initialSize) :
    bufferSize(initialSize)
{
    buffer = (T*)malloc(bufferSize*sizeof(T));
    if (buffer == NULL)
    {
        throw std::runtime_error("Out of memory");
    }
}

template <typename T> 
StaticContainer<T>::StaticContainer()
{
    StaticContainer(__SC_DEFAULT_BUFFER_SIZE);
}

template <typename T> 
StaticContainer<T>::~StaticContainer()
{
    free(buffer);
}

template <typename T> 
T* StaticContainer<T>::next()
{
    if (nextElement >= buffer + bufferSize)
    {
        throw StaticContainerException(0);
    }
    return nextElement++;
}

template <typename T> 
template <typename I>
T* StaticContainer<T>::setNext(I i)
{
    T* nextp = next();
    nextp->set(i);
    return nextp;
}

template <typename T> 
T* StaticContainer<T>::push_back(const T* e)
{
    return (T*)memcpy(next(),e,sizeof(T));
}

template <typename T> 
void StaticContainer<T>::pop_back()
{
    --nextElement;
}

template <typename T> 
void StaticContainer<T>::clear()
{
    nextElement = buffer;
}


template <typename T> 
typename StaticContainer<T>::iterator StaticContainer<T>::begin() 
{ 
    return buffer; 
}

template <typename T> 
typename StaticContainer<T>::iterator StaticContainer<T>::end() 
{ 
    return nextElement; 
}

template <typename T> 
typename StaticContainer<T>::iterator StaticContainer<T>::last() 
{ 
    return nextElement-1; 
}

template <typename T> 
typename StaticContainer<T>::reference StaticContainer<T>::operator[] (size_t n)
{
    return buffer + n;
}

#endif
