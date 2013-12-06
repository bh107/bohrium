#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
const char* enumstr_to_ctypestr(const char* enumstr)
{
    if (false) {}
    %for $enumstr, $ctypestr in $types
    else if (strcmp("$enumstr", enumstr)==0) { return "$ctypestr"; }
    %end for
    else { return "{{UNKNOWN}}"; }
}

