#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
const char* enum_to_ctypestr(bh_type type)
{
    switch(type) {
        %for $bhtype, $cpp in $types
        case $bhtype: return "$cpp";
        %end for

        default:
            return "{{UNKNOWN}}";
    }
}

