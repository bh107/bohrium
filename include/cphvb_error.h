/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
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

#ifndef __CPHVB_ERROR_H
#define __CPHVB_ERROR_H

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes */
enum /* cphvb_error */
{
    CPHVB_SUCCESS,               // General success
    CPHVB_ERROR,                 // Fatal error 
    CPHVB_TYPE_NOT_SUPPORTED,    // Data type not supported
    CPHVB_OUT_OF_MEMORY,         // Out of memory
    CPHVB_PARTIAL_SUCCESS,       // Recoverable
    CPHVB_INST_PENDING,           // Instruction is not executed
    CPHVB_INST_NOT_SUPPORTED,    // Instruction not supported
    CPHVB_USERFUNC_NOT_SUPPORTED,// User-defined function not supported
};


#ifdef __cplusplus
}
#endif

#endif
