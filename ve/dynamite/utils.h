#include "bh.h"
#ifndef __BH_VE_DYNAMITE_UTILS_H
#define __BH_VE_DYNAMITE_UTILS_H

size_t read_file(const char* filename, char** contents);
void assign_string(char*& output, const char* input);
const char* type_text(bh_type type);

const char* bhtype_to_ctype(bh_type type);
const char* bhtype_to_shorthand(bh_type type);
const char* bhopcode_to_csrc(bh_opcode opc);

#endif

