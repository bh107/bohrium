#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
const char* enum_to_ctypestr(bh_type type)
{
    switch(type) {
        %for $bhtype, $c in $types
        case $bhtype: return "$c";
        %end for

        default:
            return "{{UNKNOWN}}";
    }
}

