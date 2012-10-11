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
#include <list>
#include <iostream>

#ifdef DEBUG
#include <typeinfo>
#endif



class StaticStoreException : public std::exception {};


template <typename T>
class StaticStore
{
  private:
    T* buffer;
    T* nextElement;
    size_t bufferSize;
    std::deque<T*> emptySlot;
    std::list<T*> allocatedBuffers;
    int counter;
  public:
    StaticStore(size_t size);
    StaticStore();
    ~StaticStore();
    void clear();
    void erase(T* e);
#if __cplusplus > 199711L
    template <typename... As>
    T* next(As... as);
#endif
    T* c_next();
};


template <typename T>
StaticStore<T>::StaticStore(size_t initialSize) :
    bufferSize(initialSize)
{
    buffer = (T*)malloc(bufferSize*sizeof(T));
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
StaticStore<T>::~StaticStore()
{
	typename std::list<T*>::iterator it;
    
    if(counter > 0)
        std::cout << "[StaticStore] Warning " << counter << " arrays were not destroyed" << std::endl;

    for (it = allocatedBuffers.begin(); it != allocatedBuffers.end(); it++)
	    free(*it);
    
    allocatedBuffers.clear();
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
    	//Out of space, create a new block
		T* tmp = (T*)malloc(bufferSize*sizeof(T));
		if (tmp == NULL)
		{
			throw std::runtime_error("Out of memory");
		}
		
#ifdef DEBUG
    std::cout << "StaticStore<" << typeid(T).name() << ">(): ";
    std::cout << "Allocated " << bufferSize*sizeof(T) << " bytes; ";
#endif

		allocatedBuffers.push_back(buffer);
		buffer = tmp;
		nextElement = buffer;

		return nextElement++;

    }
}

#if __cplusplus > 199711L
template <typename T>
template <typename... As>
T* StaticStore<T>::next(As... as)
{
	return new(c_next()) T(as...);
}
#endif

template <typename T>
void StaticStore<T>::clear()
{
	typename std::list<T*>::iterator it;

#ifdef DEBUG
	if (counter != 0)
	{
		std::cout << "StaticStore<" << typeid(T).name() << ">(): ";
		std::cout << "Warning, clear called while counter was: " << counter << "; ";
    }
#endif

    counter = 0;
    nextElement = buffer;
    
    for (it = allocatedBuffers.begin(); it != allocatedBuffers.end(); it++)
	    free(*it);
    
    allocatedBuffers.clear();
    emptySlot.clear();
}

template <typename T>
void StaticStore<T>::erase(T* e)
{
    counter--;
    emptySlot.push_back(e);
}

#endif
