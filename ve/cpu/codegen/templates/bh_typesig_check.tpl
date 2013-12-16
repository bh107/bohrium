#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
bool bh_typesig_check(int typesig)
{
    switch(typesig) {
        %for $typesig, $shorthand, $descr in $cases
        case $typesig: return true; // $shorthand: $descr
        %end for

        default:
            return false;
    }
}

