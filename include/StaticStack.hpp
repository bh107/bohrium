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

#ifndef __STATICSTACK_HPP
#define __STATICSTACK_HPP

#include <cstring>
#include <vector>
#include <stdexcept>
#include <cstdlib>
#include <iostream>

#ifdef DEBUG
#include <iostream>
#include <typeinfo> 
#endif


#define __SC_DEFAULT_BUFFER_SIZE (4096)

class StaticStackException 
{
private:
    int code;
public:
    StaticStackException(int code_) : code(code_) {}
};

template <typename T>
class StaticStack
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
    StaticStack(size_t size);
    StaticStack();
    ~StaticStack();
    virtual T* push_back(const T& e);
    virtual void pop_back();
    virtual void clear();
    template <typename... As>
    T* next(As... as);
    virtual iterator begin();
    virtual iterator end();
    virtual iterator last();
    //virtual reference at(size_t n);
    virtual reference operator[] (size_t n);
    virtual size_t size();
};


template <typename T> 
StaticStack<T>::StaticStack(size_t initialSize) :
    bufferSize(initialSize)
{
    buffer = (T*)malloc(initialSize*sizeof(T));
    if (buffer == NULL)
    {
        throw std::runtime_error("Out of memory");
    }
    nextElement = buffer;
#ifdef DEBUG
    std::cout << "StaticStack<" << typeid(T).name() << ">(): ";
    std::cout << "\n  buffer: " << buffer;
    std::cout << "\n  bufferSize: " << bufferSize;
    std::cout << "\n  dataSize: " << bufferSize*sizeof(T) << std::endl;
#endif
}

template <typename T> 
StaticStack<T>::StaticStack()
{
    StaticStack(__SC_DEFAULT_BUFFER_SIZE);
}

template <typename T> 
StaticStack<T>::~StaticStack()
{
    free(buffer);
}

template <typename T> 
template <typename... As>
T* StaticStack<T>::next(As... as)
{
    if (nextElement >= buffer + bufferSize)
    {
        throw StaticStackException(0);
    }
    return new(nextElement++) T(as...);
}

template <typename T> 
T* StaticStack<T>::push_back(const T& e)
{
    return new(nextElement++) T(e);
}

template <typename T> 
void StaticStack<T>::pop_back()
{
    --nextElement;
}

template <typename T> 
void StaticStack<T>::clear()
{
    nextElement = buffer;
}

template <typename T> 
size_t StaticStack<T>::size()
{
    return (buffer - nextElement) / sizeof(T);
}

template <typename T> 
typename StaticStack<T>::iterator StaticStack<T>::begin() 
{ 
    return buffer; 
}

template <typename T> 
typename StaticStack<T>::iterator StaticStack<T>::end() 
{ 
    return nextElement; 
}

template <typename T> 
typename StaticStack<T>::iterator StaticStack<T>::last() 
{ 
    return nextElement-1; 
}

template <typename T> 
typename StaticStack<T>::reference StaticStack<T>::operator[] (size_t n)
{
    return buffer + n;
}

#endif
