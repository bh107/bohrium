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

#ifndef __BH_SEQSET_HPP
#define __BH_SEQSET_HPP

#include <map>
#include <stdexcept>

/* Container class that works like a set (i.e. all objects are unique)
 * while also maintaining the insertion order.
 */
template <typename T>
class seqset
{
private:
    size_t maxid;
    std::map<T,size_t> _map;
    std::map<size_t, T> _rmap;
public:
    seqset() : maxid(0) {}

    // Clear the containor
    void clear()
    {
        maxid = 0;
        _map.clear();
        _rmap.clear();
    }

    // Insert an object
    std::pair<size_t,bool> insert(const T &v)
    {
        auto it = _map.find(v);
        if (it == _map.end())
        {
            size_t id = maxid++;
            _map.insert(std::make_pair(v,id));
            _rmap.insert(std::make_pair(id,v));
            return std::make_pair(id,true);

        } else {
            return  std::make_pair(it->second,false);
        }
    }

    size_t erase(const T &v)
    {
        auto it = _map.find(v);
        if (it != _map.end())
        {
            _rmap.erase(it->second);
            _map.erase(it);
            return 1;
        } else
            return 0;
    }

    // Get the id of an object
    size_t operator[] (const T &v) const
    {
        auto it = _map.find(v);
        if (it == _map.end())
            throw std::out_of_range("Object unknown");
        return it->second;
    }

    // Get an object given the id
    const T& operator[] (size_t id) const
    {
        auto it = _rmap.find(id);
        if (it == _rmap.end())
            throw std::out_of_range("Object unknown");
        return it->second;
    }

    typename std::map<size_t, T>::const_iterator begin() const
    {
        return _rmap.cbegin();
    }

    typename std::map<size_t, T>::const_iterator end() const
    {
        return _rmap.cend();
    }

    size_t size() const
    {
        return _map.size();
    }

    std::set<T> set() const
    {
        std::set<T> res;
        for (const std::pair<T,size_t>& p: _map)
            res.insert(p.first);
        return res;
    }
};

#endif
