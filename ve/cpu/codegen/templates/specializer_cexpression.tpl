#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
#include "specializer.hpp"

using namespace std;
namespace bohrium {
namespace engine {
namespace cpu {

string Specializer::cexpression(Block& block, size_t tac_idx)
{
    tac_t& tac  = block.program[tac_idx];
    ETYPE etype = block.scope[tac.out].type;

    switch(tac.oper) {
    
    }

    return "__ERR_OPER__";
}    

}}}
