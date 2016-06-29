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

#include <sstream>
#include <cassert>

#include "block.hpp"

using namespace std;

namespace bohrium {

namespace {
void spaces(stringstream &out, int num) {
    for (int i = 0; i < num; ++i) {
        out << " ";
    }
}
}

Block* Block::findInstrBlock(const bh_instruction *instr) {
    if (_block_list.size() == 0) {
        assert(_instr != NULL);
        if (_instr == instr)
            return this;
    } else {
        for(Block &b: this->_block_list) {
            Block *ret = b.findInstrBlock(instr);
            if (ret != NULL)
                return ret;
        }
    }
    return NULL;
}

string Block::pprint() const {
    stringstream ss;
    spaces(ss, rank*4);
    if (_instr != NULL) {
        assert(_block_list.size() == 0);
        ss << *_instr << endl;
    } else {
        ss << "rank: " << rank << ", size: " << size;
        if (_sweeps.size() > 0) {
            ss << ", sweeps: { ";
            for (const bh_instruction *b : _sweeps) {
                ss << *b << ",";
            }
            ss << "}";
        }
        if (_block_list.size() > 0) {
            ss << ", block list:" << endl;
            for (const Block &b : _block_list) {
                ss << b.pprint() << endl;
            }
        }
    }
    return ss.str();
}

void Block::getAllInstr(vector<const bh_instruction *> &out) const {
    if (_instr != NULL) {
        assert(_block_list.size() == 0);
        out.push_back(_instr);
    } else {
        for (const Block &b : _block_list) {
            b.getAllInstr(out);
        }
    }
}

vector<const bh_instruction *> Block::getAllInstr() const {
    vector<const bh_instruction *> ret;
    getAllInstr(ret);
    return ret;
}

void Block::merge(const Block &b) {
    assert(_instr == NULL);
    assert(b._instr == NULL);
    _block_list.insert(_block_list.end(), b._block_list.begin(), b._block_list.end());
    _sweeps.insert(b._sweeps.begin(), b._sweeps.end());
}

ostream& operator<<(ostream& out, const Block& b) {
    out << b.pprint();
    return out;
}

ostream& operator<<(ostream& out, const vector<Block>& block_list) {
    out << "Block list: " << endl;
    for(const Block &b: block_list) {
        out << b;
    }
    return out;
}


} // bohrium
