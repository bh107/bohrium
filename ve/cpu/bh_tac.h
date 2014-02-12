typedef enum OPERATION {
    MAP         = 1,    // aka elementwise operation unary operator
    ZIP         = 2,    // aka elementwise operation with binary operator
    REDUCE      = 4,
    SCAN        = 8,
    GENERATE    = 16,   // examples: iota, random numbers etc.
    SYSTEM      = 32,
    EXTENSION   = 64,
} OPERATION;

typedef enum OPERATOR {
    // Used by elementwise operations
    ABSOLUTE,
    ARCCOS,
    ARCCOSH,
    ARCSIN,
    ARCSINH,
    ARCTAN,
    ARCTANH,
    CEIL,
    COS,
    COSH,
    EXP,
    EXP2,
    EXPM1,
    FLOOR,
    IDENTITY,
    IMAG,
    INVERT,
    ISINF,
    ISNAN,
    LOG,
    LOG10,
    LOG1P,
    LOG2,
    LOGICAL_NOT,
    REAL,
    RINT,
    SIN,
    SINH,
    SQRT,
    TAN,
    TANH,
    TRUNC,

    // Used by elementwise, reduce, and scan operations
    ADD,
    ARCTAN2,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_XOR,
    DIVIDE,
    EQUAL,
    GREATER,
    GREATER_EQUAL,
    LEFT_SHIFT,
    LESS,
    LESS_EQUAL,
    LOGICAL_AND,
    LOGICAL_OR,
    LOGICAL_XOR,
    MAXIMUM,
    MINIMUM,
    MOD,
    MULTIPLY,
    NOT_EQUAL,
    POWER,
    RIGHT_SHIFT,
    SUBTRACT,

    // Used by system operations
    DISCARD,
    FREE,
    NONE,
    SYNC,

    // Used by generator operations
    FLOOD,
    RANDOM,
    RANGE,

    NBUILTIN,   // Not an operator but a count of built-in operators
    EXT_OFFSET  // Wildcard for extension opcodes

} OPERATOR;

typedef struct tac {
    OPERATION op;       // Operation
    OPERATOR  oper;     // Operator
    uint32_t  out;      // Output operand
    uint32_t  in1;      // First input operand
    uint32_t  in2;      // Second input operand
} tac_t;

typedef enum LAYOUT {
    CONSTANT,
    CONTIGUOUS,
    STRIDED,
    SPARSE
} LAYOUT;

typedef enum LMASK {
    LMASK_CC,
    LMASK_CK,
    LMASK_CCC,
    LMASK_CKC,
    LMASK_CCK
} LMASK;

//
// NOTE: Changes to bk_kernel_args_t must be 
//       replicated to "templates/kernel.tpl".
//
typedef struct block_arg {
    LAYOUT  layout;     // The layout of the data
    void*   data;       // Pointer to memory allocated for the array
    int64_t type;       // Type of the elements stored
    int64_t start;      // Offset from memory allocation to start of array
    int64_t nelem;      // Number of elements available in the allocation

    int64_t ndim;       // Number of dimensions of the array
    int64_t* shape;     // Shape of the array
    int64_t* stride;    // Stride in each dimension of the array
} block_arg_t;      // Meta-data for a kernel argument

#define HAS_ARRAY_OP MAP | ZIP | REDUCE | SCAN | GENERATE

