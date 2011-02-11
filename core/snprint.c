/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include "cphvb.h"

#define CPHVB_TYPE -1

int _print_array(void* arr, int e, cphvb_type type,
                   size_t size, char* str)
{
    int res = 0;
    int nc, i;
    if (e > 0) 
    {
        switch(type)
        {
        case CPHVB_INT32:
            nc = snprintf(str, size, "%d",((cphvb_int32*)arr)[0]);
            break;
        case CPHVB_INT64:
            nc = snprintf(str, size, "%ld",((cphvb_int64*)arr)[0]);
            break;
        case CPHVB_TYPE:
            nc = snprintf(str, size, "%s",
                          cphvb_type_text(((cphvb_type*)arr)[0]));
            break;
        default:
            nc = 0;
        }
        str += nc; res += nc; size -= nc;
    }
    for (i = 1; i < e; i++)
    {
        switch(type)
        {
        case CPHVB_INT32:
            nc = snprintf(str, size, ", %d",((cphvb_int32*)arr)[i]);
            break;
        case CPHVB_INT64:
            nc = snprintf(str, size, ", %ld",((cphvb_int64*)arr)[i]);
            break;
        case CPHVB_TYPE:
            nc = snprintf(str, size, ", %s",
                          cphvb_type_text(((cphvb_type*)arr)[i]));
            break;
        default:
            nc = 0;
        }
        str += nc; res += nc; size -= nc;    
    }
    return res;
}

int _print_array_array(void** arr, int e1, int e2, cphvb_type type,
                       size_t size, char* str)
{
    int res = 0;
    int nc, i;
    for(i = 0; i < e1; ++i)
    {
        nc = snprintf(str, size, "(");
        str += nc; res += nc; size -= nc;
        nc = _print_array(arr[i],e2,type,size,str);
        str += nc; res += nc; size -= nc;
        nc = snprintf(str, size, ")");
        str += nc; res += nc; size -= nc;
    }
    return res;
}

/* Pretty print instruktion
 *
 * Mainly for debugging purposes
 *
 * @inst   Instruction to print
 * @size   Write at most this many bytes to buffer
 * @buf    Buffer to contain the string
 */
int cphvb_snprint(const cphvb_instruction *inst , size_t size, char* str)
{
    int res = 0;
    int nc, i;
    int nops = cphvb_operands(inst->opcode);
    nc = snprintf(str, size, "{\n\topcode: %s\n\tndim: %d\n\toperand: (", 
                  cphvb_opcode_text(inst->opcode), inst->ndim);
    str += nc; res += nc; size -= nc;
    nc = _print_array(inst->operand,nops,CPHVB_INT32,size,str);
    str += nc; res += nc; size -= nc;
    nc = snprintf(str, size, ")\n\ttype: (");
    str += nc; res += nc; size -= nc;
    nc = _print_array(inst->type,nops,CPHVB_TYPE,size,str);
    str += nc; res += nc; size -= nc;
    nc = snprintf(str, size, ")\n\tshape: (");
    str += nc; res += nc; size -= nc;
    nc = _print_array(inst->shape,inst->ndim,CPHVB_INT64,size,str);
    str += nc; res += nc; size -= nc;
    nc = snprintf(str, size, ")\n\tstart: (");
    str += nc; res += nc; size -= nc;
    nc = _print_array(inst->start,nops,CPHVB_INT64,size,str);
    str += nc; res += nc; size -= nc;
    nc = snprintf(str, size, ")\n\tstride: (");
    str += nc; res += nc; size -= nc;
    nc = _print_array_array((void**)inst->stride,nops,inst->ndim,CPHVB_INT64,size,str);
    str += nc; res += nc; size -= nc;
    nc = snprintf(str, size, ")\n\tconstant: (");
    str += nc; res += nc; size -= nc;
    int cidx = 0; //constant[] index
    for(i = 0; i < nops; i++)
    {
        if(inst->operand[i] == CPHVB_CONSTANT)
        { 
            switch(inst->type[i])
            {
            case CPHVB_INT8:
                nc = snprintf(str, size, "(%hhi)",inst->constant[cidx].int8);
                break;
            case CPHVB_INT16:
                nc = snprintf(str, size, "(%hi)",inst->constant[cidx].int16);
                break;
            case CPHVB_INT32:
                nc = snprintf(str, size, "(%i)",inst->constant[cidx].int32);
                break;
            case CPHVB_INT64:
                nc = snprintf(str, size, "(%lli)",
                              (long long int)inst->constant[cidx].int64);
                break;
            case CPHVB_UINT8:
                nc = snprintf(str, size, "(%hhu)",inst->constant[cidx].uint8);
                break;
            case CPHVB_UINT16:
                nc = snprintf(str, size, "(%hu)",inst->constant[cidx].uint16);
                break;
            case CPHVB_UINT32: 
                nc = snprintf(str, size, "(%u)",inst->constant[cidx].uint32);
                break;
            case CPHVB_UINT64:
                nc = snprintf(str, size, "(%llu)",
                              (long long int)inst->constant[cidx].uint64);
                break;
/*            case CPHVB_FLOAT16:
                nc = snprintf(str, size, "(%f)",inst->constant[cidx].float16);
                break;
*/            case CPHVB_FLOAT32:
                nc = snprintf(str, size, "(%f)",inst->constant[cidx].float32);
                break;
            case CPHVB_FLOAT64:
                nc = snprintf(str, size, "(%f)",inst->constant[cidx].float64);
                break;
            case CPHVB_PTR:
                nc = snprintf(str, size, "(%p)",inst->constant[cidx].ptr);
                break;
            default:
                nc = snprintf(str, size, "(unknown type: %d)",inst->type[i]);
            }
            str += nc; res += nc; size -= nc; ++cidx;
        } 
    }
    nc = snprintf(str, size, ")\n}\n");
    str += nc; res += nc; size -= nc;
    return res;
}
