#include <sstream>
#include "dag.hpp"
#include "symbol_table.hpp"
#include "utils.hpp"

//
// Mostly boiler-plate code, (de)constructor, getters, etc.
//

using namespace std;
using namespace boost;
namespace bohrium{
namespace core {

Dag::Dag(bh_instruction* instr, bh_intp ninstr) : instr_(instr), ninstr_(ninstr), symbol_table_(ninstr*6+2), tacs_(ninstr), graph_(ninstr), subgraphs_()
{
    DEBUG(TAG,"Dag(...)");

    //
    // Map instructions to tac and construct symbol-table.
    instrs_to_tacs(instr, ninstr, tacs_, symbol_table_);

    // Construct dependencies based on array operations
    array_deps();
    // Construction dependencies based on system operations
    system_deps();

    // Construct subgraphs
    partition();

    DEBUG(TAG,"Dag(...);");
}

Dag::~Dag(void)
{
}

tac_t& Dag::tac(size_t tac_idx)
{
    return tacs_[tac_idx];
}

bh_instruction& Dag::instr(size_t instr_idx)
{
    return instr_[instr_idx];
}

vector<Graph*>& Dag::subgraphs(void)
{
    return subgraphs_;
}

}}
