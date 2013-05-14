#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
inline
const char* bhopcode_to_cexpr(bh_opcode opcode)
{
    switch(instr->opcode) {
        /* For now only element-wise wise

        // System (memory and stuff)
        %for $opcode in $system
        case $opcode["opcode"]:
            return "${opcode["code"]}";
        %end for

        // Extensions (ufuncs)
        %for $opcode in $extensions
        case $opcode["opcode"]:
            return "${opcode["code"]}";
        %end for

        // Partial Reductions
        %for $opcode in $reductions
        case $opcode["opcode"]:
            return "${opcode["code"]}";
        %end for
        */

        // Binary elementwise: ADD, MULTIPLY...
        %for $opcode in $binary
        case $opcode["opcode"]:
            return "${opcode["code"]}";
        %end for

        // Unary elementwise: SQRT, SIN...
        %for $opcode in $unary
        case $opcode["opcode"]:
            return "$opcode["code"]";
        %end for

        case BH_UNKNOWN:
            return "BH_UNKNOWN";
        default:
            return "Unknown type";

    }

}

