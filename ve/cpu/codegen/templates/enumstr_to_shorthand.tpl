#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
const char* enumstr_to_shorthand(const char* enumstr)
{
    if (false) {}
    %for $enumstr, $shorthand in $types
    else if (strcmp("$enumstr", enumstr)==0) { return "$shorthand"; }
    %end for
    else { return "{{UNKNOWN}}"; }
}

