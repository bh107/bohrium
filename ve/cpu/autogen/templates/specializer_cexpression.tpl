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
    string expr_text;

    switch(tac.oper) {
    %for $oper, $op_and_etype_expr in $expressions
        case $oper:            
            %if len($op_and_etype_expr) > 1
            switch (tac.op) {
                %for $op, $etype_expr in $op_and_etype_expr
                case $op:
                    %if len($etype_expr) > 1
                    switch(etype) {
                        %for $etype, $expr in $etype_expr
                        %if $etype == "default"
                        %set $case = ""
                        %else
                        %set $case = "case "
                        %end if
                        //$case$etype: return $expr;
                        $case$etype: expr_text = "$expr"; break;
                        %end for
                    }
                    %else 
                    //return $etype_expr[0][1];
                    expr_text = "$etype_expr[0][1]"; break;
                    %end if             
                %end for
                default:
                    //return "__ERR_UNS_OPER__";
                    expr_text = "__ERR_UNS_OPER__"; break;
            }            
            %else            
            %set $op, $etype_expr = $op_and_etype_expr[0]
            %if len($etype_expr) > 1
            switch(etype) {
                %for $etype, $expr in $etype_expr
                %if $etype == "default"
                %set $case = ""
                %else
                %set $case = "case "
                %end if
                //$case$etype: return $expr;
                $case$etype: expr_text = "$expr"; break;
                %end for
            }
            %else 
            //return $etype_expr[0][1];
            expr_text = "$etype_expr[0][1]"; break;
            %end if
            %end if
            break;
    %end for
    }

    switch(utils::tac_noperands(tac)) {
        case 3:
            return utils::string_format(expr_text, tac.out, tac.in1, tac.in2);
        case 2:
            return utils::string_format(expr_text, tac.out, tac.in1);
        case 1:
            return utils::string_format(expr_text, tac.out);
        default:
            return expr_text;
    }
}    

}}}
