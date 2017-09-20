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
#include "contracter.hpp"

using namespace std;

namespace bohrium {
namespace filter {
namespace bccon {
    bool __verbose = false;

Contracter::Contracter(
    bool verbose,
    bool repeats,
    bool reduction,
    bool stupidmath,
    bool collect,
    bool muladd)
    : repeats_(repeats),
      reduction_(reduction),
      stupidmath_(stupidmath),
      collect_(collect),
      muladd_(muladd) {
            __verbose = verbose;
      }

Contracter::~Contracter(void) {}

void Contracter::contract(BhIR& bhir)
{
    if(reduction_)  contract_reduction(bhir);
    if(stupidmath_) contract_stupidmath(bhir);
    if(collect_)    contract_collect(bhir);
    if(muladd_)     contract_muladd(bhir);
    if(repeats_)    contract_repeats(bhir);
}

void verbose_print(std::string str)
{
    if (__verbose) {
        std::cout << "[Contracter] " << str << std::endl;
    }
}

}}}
