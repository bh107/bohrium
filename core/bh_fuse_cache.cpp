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
#include <exception>
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

    // Constructor of the InstrHash class
    InstrHash::InstrHash(BatchHash &batch, const bh_instruction &instr)
    {
        /* The Instruction hash consists of the following fields:
         * <opcode> (<operant-id> <ndim> <shape>)[1] <sweep-dim>[2] <seperator>
         * 1: for each operand
         * 2: if the operation is a sweep operation
         */
        int noperands = bh_operands(instr.opcode);
        this->append((char*)&instr.opcode, sizeof(instr.opcode));               // <opcode>
        for(int oidx=0; oidx<noperands; ++oidx) {
            const bh_view& view = instr.operand[oidx];
            if (bh_is_constant(&view))
                continue;  // Ignore constants
            std::pair<size_t,bool> vid = batch.views.insert(view);
            size_t id = vid.first;
            this->append((char*)&id, sizeof(id));                               // <operant-id>
            this->append((char*)&view.ndim, sizeof(view.ndim));                 // <ndim>
            this->append((char*)&view.shape, sizeof(bh_index)*view.ndim);       // <shape>
        }
        if (bh_opcode_is_sweep(instr.opcode))
            this->append((char*)&instr.constant.value.int64, sizeof(bh_int64)); // <sweep-dim>
        const size_t sep = SIZE_MAX;
        this->append((char*)&sep, sizeof(sep));                                 // <separator>
    }

    //Constructor of the BatchHash class
    BatchHash::BatchHash(const vector<bh_instruction> &instr_list)
    {
        string data;
        BOOST_FOREACH(const bh_instruction &instr, instr_list)
        {
            data.append(InstrHash(*this, instr));
        }
        boost::hash<string> hasher;
        _hash = hasher(data);
    }

    InstrIndexesList &FuseCache::insert(const BatchHash &batch,
                                        const vector<bh_ir_kernel> &kernel_list)
    {
        cache[batch.hash()] = InstrIndexesList(kernel_list, batch.hash(), fuser_name);
        return cache[batch.hash()];
    }

    bool FuseCache::lookup(const BatchHash &batch,
                           bh_ir &bhir,
                           vector<bh_ir_kernel> &kernel_list) const
    {
        assert(kernel_list.size() == 0);
        CacheMap::const_iterator it = cache.find(batch.hash());
        if(deactivated or it == cache.end())
        {
            return false;
        }
        else
        {
            it->second.fill_kernel_list(bhir, kernel_list);
          return true;
        }
    }

    void FuseCache::write_to_files() const
    {
        if(deactivated)
            return;
        if(dir_path == NULL)
        {
            cout << "[FUSE-CACHE] Couldn't find the 'cache_path' key in "\
            "the configure file thus no cache files are written to disk!" << endl;
            return;
        }
        path cache_dir(dir_path);
        if(create_directories(cache_dir))
        {
            cout << "[FUSE-CACHE] Creating cache diretory " << cache_dir << endl;
        #if BOOST_VERSION > 104900
            permissions(cache_dir, all_all);
        #endif
        }

        path tmp_dir = cache_dir / unique_path();
        create_directories(tmp_dir);
        for(CacheMap::const_iterator it=cache.begin(); it != cache.end(); ++it)
        {
            string name;
            it->second.get_filename(name);
            path shared_name = cache_dir / name;

            if(exists(shared_name))
                continue;//No need to overwrite an existing file

            path unique_name = tmp_dir / name;
            ofstream ofs(unique_name.string().c_str());
            boost::archive::text_oarchive oa(ofs);
            oa << it->second;
            ofs.close();
        #if BOOST_VERSION > 104900
            permissions(unique_name, all_all);
        #endif
            rename(unique_name, shared_name);
        }
        remove(tmp_dir);
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
                int tries = 0;
                while(1)
                {
                    try
                    {
                        ifstream ifs(f.string().c_str());
                        boost::archive::text_iarchive ia(ifs);
                        InstrIndexesList t;
                        ia >> t;
                        if(iequals(t.fuser_name(), fuser_name) and
                           iequals(t.fuse_model(), fuse_model_name))
                        {
                            if(cache.find(t.hash()) != cache.end())
                            {
                                if(cache[t.hash()].cost() < t.cost())
                                {
                                    cout << "[FUSE-CACHE] ignoring cache file with higher cost" << endl;
                                    break;
                                }
                            }
                            cache[t.hash()] = t;
                        }
                    }
                    catch(const std::exception &e)
                    {
                        if(++tries >= 10)
                        {
                            cerr << "[FUSE-CACHE] failed to open file '" << f.string();
                            cerr << "' (" << tries << " tries): " << e.what() << endl;
                        }
                        else
                            continue;
                    }
                    break;
                }
            }
        }
    }
} //namespace bohrium
