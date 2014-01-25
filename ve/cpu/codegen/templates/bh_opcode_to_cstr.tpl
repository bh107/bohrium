#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
const char* bh_opcode_to_cstr(bh_opcode const opcode)
{
    switch(opcode) {
        %for $opcode, $long_str, $short_str in $opcodes
        case $opcode: return "$long_str";
        %end for

        default:
            return "{{UNKNOWN}}";
    }
}

