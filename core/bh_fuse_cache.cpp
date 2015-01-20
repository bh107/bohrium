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

#include <string>
#include <cstring>
#include <bh.h>
#include "bh_fuse.h"
#include "bh_fuse_cache.h"
#include <fstream>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/version.hpp>
#include <boost/algorithm/string/predicate.hpp> //For iequals()

using namespace std;
using namespace boost;
using namespace boost::filesystem;

namespace bohrium {

    //Constructor of the InstrHash class
    InstrHash::InstrHash(BatchHash &batch, const bh_instruction &instr)
    {
        BOOST_FOREACH(const bh_view &view, instr.operand)
        {
            if(bh_is_constant(&view))//We ignore constants
                continue;

            //Hash the base array pointer
            uint64_t base_id;
            map<const bh_base*, uint64_t>::iterator it = batch.base2id.find(view.base);
            if(it != batch.base2id.end())
            {
                base_id = it->second;
            }
            else
            {
                base_id = batch.base_id_count++;
                batch.base2id.insert(make_pair(view.base, base_id));
            }
            this->append((char*)&base_id, sizeof(base_id));

            //Hash ndim and start
            this->append((char*)&view.ndim, sizeof(view.ndim));
            this->append((char*)&view.start, sizeof(view.start));

            //Hash shape and stride
            this->append((char*)view.shape, sizeof(bh_index)*view.ndim);
            this->append((char*)view.stride, sizeof(bh_index)*view.ndim);
        }
    }

    //Constructor of the BatchHash class
    BatchHash::BatchHash(const vector<bh_instruction> &instr_list):base_id_count(0)
    {
        string data;
        BOOST_FOREACH(const bh_instruction &instr, instr_list)
        {
            data.append(InstrHash(*this, instr));
        }
        boost::hash<string> hasher;
        hash_key = hasher(data);
    }

    void FuseCache::insert(const BatchHash &batch,
                           const vector<bh_ir_kernel> &kernel_list)
    {
        cache[batch.hash()] = InstrIndexesList(kernel_list, batch.hash(), fuser_name);
    }

    bool FuseCache::lookup(const BatchHash &batch,
                           bh_ir &bhir,
                           vector<bh_ir_kernel> &kernel_list) const
    {
//        cout << "looking up " << batch.hash() << ": ";

        assert(kernel_list.size() == 0);
        CacheMap::const_iterator it = cache.find(batch.hash());
        if(it == cache.end())
        {
//            cout << "cache miss!" << endl;
            return false;
        }
        else
        {
            it->second.fill_kernel_list(bhir, kernel_list);
//          cout << "cache hit!" << endl;
          return true;
        }
    }

    void FuseCache::write_to_files() const
    {
        if(dir_path == NULL)
        {
            cout << "[FUSE-CACHE] Couldn't find the 'cache_path' key in "\
            "the configure file thus no cache files are written to disk!" << endl;
            return;
        }
        path p(dir_path);
        if(create_directories(p))
        {
            cout << "[FUSE-CACHE] Creating cache diretory " << p << endl;
        #if BOOST_VERSION > 104900
            permissions(p, all_all);
        #endif
        }

        for(CacheMap::const_iterator it=cache.begin(); it != cache.end(); ++it)
        {
            string name;
            it->second.get_filename(name);
            path filename = p / name;
            ofstream ofs(filename.string());
            boost::archive::text_oarchive oa(ofs);
            oa << it->second;
            ofs.flush();
        #if BOOST_VERSION > 104900
            permissions(filename, all_all);
        #endif
            assert(it->second.fuse_model() == model_name);
        }
    }

    void FuseCache::load_from_files()
    {
        if(dir_path == NULL)
        {
            cout << "[FUSE-CACHE] Couldn't find the 'cache_path' key in "\
            "the configure file thus no cache files are loaded from disk!" << endl;
            return;
        }
        path p(dir_path);
        if(not (exists(p) and is_directory(p)))
            return;

        string fuse_model_name;
        fuse_model_text(fuse_get_selected_model(), fuse_model_name);

        //Iterate the 'dir_path' diretory and load each file
        directory_iterator it(p), eod;
        BOOST_FOREACH(const path &f, make_pair(it, eod))
        {
            if(is_regular_file(f))
            {
                ifstream ifs(f.string());
                boost::archive::text_iarchive ia(ifs);
                InstrIndexesList t;
                ia >> t;
                if(iequals(t.fuser_name(), fuser_name) and
                   iequals(t.fuse_model(), fuse_model_name))
                {
                    cache[t.hash()] = t;
                }
            }
        }
    }
} //namespace bohrium
