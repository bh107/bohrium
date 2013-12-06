#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
inline
const char* bh_typesig_to_shorthand(int typesig)
{
    switch(typesig) {
        %for $nsig, $hsig, $tsig in $typesigs
        case $nsig: return "$hsig"; // $tsig
        %end for
        default:
            return "<UNKNOWN>";
    }
}

