#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
const char* bhtype_to_shorthand(bh_type type)
{
    switch(type) {
        %for $bhtype, $shorthand in $types
        case $bhtype: return "$shorthand";
        %end for

        default:
            return "{{UNKNOWN}}";
    }
}

