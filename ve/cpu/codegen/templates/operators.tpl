#compiler-settings
directiveStartToken= %
#end compiler-settings
%slurp
%set $sep = ""
typedef enum OPERATOR {
    // Used by elementwise, reduce and scan operations
    %for $oper in $unary
    $oper,
    %end for

    // Used by elementwise, reduce and scan operations
    %for $oper in $binary
    $oper,
    %end for

    // Used by system operations
    %for $oper in $system
    $oper,
    %end for

    // Used by generator operations
    %for $oper in $generators
    $oper,
    %end for
    
    NBUILTIN,   // Not an operator but a count of built-in operators
    USERDEF     // Wildcard for userdefined operators

} OPERATOR;

