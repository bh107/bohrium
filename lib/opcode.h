#ifndef __CPHVB_OPCODE_H
#define __CPHVB_OPCODE_H

/* Codes for known oparations */
enum cphvb_opcode
{
    CPHVB_ADD,
    CPHVB_SUB,
    CPHVB_MULT,
    CPHVB_DIV,
    CPHVB_MOD,
    CPHVB_NEG,
    CPHVB_MALLOC,
    CPHVB_FREE,
    CPHVB_READ,
    CPHVB_WRITE
};

/* Number of operands for a given operation */
static const int cphvb_operands[] = 
{ 
    [CPHVB_ADD] = 3,
    [CPHVB_SUB] = 3,
    [CPHVB_MULT] = 3,
    [CPHVB_DIV] = 3,
    [CPHVB_MOD] = 3,
    [CPHVB_NEG] = 2,
    [CPHVB_MALLOC] = 1,
    [CPHVB_FREE] = 1,
    [CPHVB_READ] = 2,
    [CPHVB_WRITE] = 2
};

#define CPHVB_MAX_NO_OPERANDS 3

#endif
