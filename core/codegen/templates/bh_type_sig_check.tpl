#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
/**
 *  Determine whether the given typesig, in the coding produced by bh_typesig, is valid.
 *
 *  @param instr The instruction for which to deduct a signature.
 *  @return The deducted signature.
 */
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

