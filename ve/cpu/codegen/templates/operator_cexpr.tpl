#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
const char* operator_cexpr(OPERATION const op, OPERATOR const oper, const bh_type type)
{
    switch(oper) {
        %for $oper, $code in $operators
        case $oper:
            return "$code";
        %end for

        default:
            return "__UNK_OPER__";
    }
}
