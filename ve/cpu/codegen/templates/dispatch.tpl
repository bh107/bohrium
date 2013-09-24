#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
void dispatch(bh_instruction* instr)
{
    switch(instr->opcode) {
        // System (memory and stuff)
        %for $opcode in $system
        case $opcode["opcode"]:
        %end for

        // Extensions (ufuncs)
        %for $opcode in $extensions
        case $opcode["opcode"]:
        %end for

        // Partial Reductions
        %for $opcode in $reductions
        case $opcode["opcode"]:
        %end for

        // Binary elementwise: ADD, MULTIPLY...
        %for $opcode in $binary
        case $opcode["opcode"]:
        %end for

        // Unary elementwise: SQRT, SIN...
        %for $opcode in $unary
        case $opcode["opcode"]:
        %end for

        case BH_UNKNOWN:
            return "BH_UNKNOWN";
        default:
            return "Unknown type";

    }
}


