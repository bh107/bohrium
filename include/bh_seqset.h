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
};


