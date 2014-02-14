#slurp
#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
std::string layoutmask_shorthand(TAC_LMASK const layoutmask)
{
    switch(layoutmask) {
    %for $masks in $lshort
        case $masks[0]:
            return "$masks[1]";
    %end for
    }
}

typedef enum TAC_LMASK {
    %for $amount in $lmasks
    %for $shorthand, $mask in $lmasks[$amount]
    %if $amount == 1
    $shorthand = ($mask[0] << 8),
    %elif $amount == 2
    $shorthand = ($mask[0] << 8) | ($mask[1] << 4),
    %elif $amount == 3
    $shorthand = ($mask[0] << 8) | ($mask[1] << 4) | $mask[2],
    %end if
    %end for
    %end for
} TAC_LMASK;    // Uses three bytes, one for each arg

