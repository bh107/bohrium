#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
inline
const char* bhtype_to_ctype(bh_type type)
{
    switch(type) {
        %for $bhenum, $shorthand, $ctype in $types
        case $bhenum:
            return "$ctype";
        %end for
        case BH_UNKNOWN:
            return "BH_UNKNOWN";
        default:
            return "Unknown type";
    }
}

inline
const char* bhtype_to_shorthand(bh_type type)
{
    switch(type) {
        %for $bhenum, $shorthand, $ctype in $types
        case $bhenum:
            return "$shorthand";
        %end for
        case BH_UNKNOWN:
            return "BH_UNKNOWN";
        default:
            return "Unknown type";
    }
}


