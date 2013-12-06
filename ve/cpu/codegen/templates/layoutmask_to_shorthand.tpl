#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%
const char* bh_layoutmask_to_shorthand(const int mask)                                                               
{
    switch(mask) {
        default:
            printf("Err: Unsupported layoutmask [%d]\n", mask);
            return "_UNS_";
    }
}

