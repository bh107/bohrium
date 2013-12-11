#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
const char* bh_layoutmask_to_shorthand(const int mask)
{
    switch(mask) {
        %for $mask_n, $mask_c in $masks
        case $mask_n: return "$mask_c"; 
        %end for

        default:
            printf("cpu(bh_layoutmask_to_shorthand): Unsupported layoutmask [%d]\n", mask);
            return "{{UNSUPPORTED}}";
    }
}

