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

Dag::Dag(SymbolTable& symbol_table, std::vector<tac_t>& program)
    : symbol_table_(symbol_table), program_(program),
      graph_(program.size()), subgraphs_()
{
    DEBUG(TAG,"Dag(...)");

    array_deps();   // Construct dependencies based on array operations
    system_deps();  // Construct dependencies based on system operations
    partition();    // Construct subgraphs

    DEBUG(TAG,"Dag(...);");
}

Dag::~Dag(void)
{
}

tac_t& Dag::tac(size_t tac_idx)
{
    return program_[tac_idx];
}

vector<Graph*>& Dag::subgraphs(void)
{
    return subgraphs_;
}

}}
