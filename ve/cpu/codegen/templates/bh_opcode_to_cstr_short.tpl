#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
const char* bh_opcode_to_cstr_short(bh_opcode const opcode)
{
    switch(opcode) {
        %for $opcode, $long_str, $short_str in $opcodes
        case $opcode: return "$short_str";
        %end for

        default:
            return "{{UNKNOWN}}";
    }
}

