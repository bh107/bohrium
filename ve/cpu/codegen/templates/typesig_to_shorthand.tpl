#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
const char* bh_typesig_to_shorthand(int typesig)
{
    switch(typesig) {

        default:
            printf("Err: Unsupported type signature %d.\n", typesig);
            return "_UNS_";
    }
}

