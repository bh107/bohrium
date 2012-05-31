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
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */

#ifdef __APPLE__
 //error.h is not required but the functions are known
 //the file is not found on OSX
#else
#include <error.h>
#endif
#include <cphvb.h>
#include <stdbool.h>

/* Text string for error code
 *
 * @error  Error code.
 * @return Text string.
 */
const char* cphvb_error_text(cphvb_error error)
{
    switch(error)
    {
    case CPHVB_SUCCESS: 
        return "CPHVB_SUCCESS";
    case CPHVB_ERROR: 
        return "CPHVB_ERROR";
    case CPHVB_TYPE_ERROR: 
        return "CPHVB_TYPE_ERROR";
    case CPHVB_TYPE_NOT_SUPPORTED: 
        return "CPHVB_TYPE_NOT_SUPPORTED";
    case CPHVB_TYPE_NOT_SUPPORTED_BY_OP: 
        return "CPHVB_TYPE_NOT_SUPPORTED_BY_OP";
    case CPHVB_TYPE_COMB_NOT_SUPPORTED: 
        return "CPHVB_TYPE_COMB_NOT_SUPPORTED";
    case CPHVB_OUT_OF_MEMORY: 
        return "CPHVB_OUT_OF_MEMORY";
    case CPHVB_RESULT_IS_CONSTANT: 
        return "CPHVB_RESULT_IS_CONSTANT";
    case CPHVB_OPERAND_UNKNOWN: 
        return "CPHVB_OPERAND_UNKNOWN";
    case CPHVB_ALREADY_INITALIZED: 
        return "CPHVB_ALREADY_INITALIZED";
    case CPHVB_NOT_INITALIZED: 
        return "CPHVB_NOT_INITALIZED";
    case CPHVB_PARTIAL_SUCCESS: 
        return "CPHVB_PARTIAL_SUCCESS";
    case CPHVB_INST_DONE:
        return "CPHVB_INST_DONE";
    case CPHVB_INST_UNDONE:
        return "CPHVB_INST_UNDONE";
    case CPHVB_INST_NOT_SUPPORTED: 
        return "CPHVB_INST_NOT_SUPPORTED";
    case CPHVB_INST_NOT_SUPPORTED_FOR_SLICE: 
        return "CPHVB_INST_NOT_SUPPORTED_FOR_SLICE";
    case CPHVB_USERFUNC_NOT_SUPPORTED: 
        return "CPHVB_USERFUNC_NOT_SUPPORTED";
    default:
        return "Error code unknown";
	}

}
