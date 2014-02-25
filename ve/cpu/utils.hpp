#ifndef __BH_VE_CPU_UTILS
#define __BH_VE_CPU_UTILS
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdarg>

#include "bh.h"
#include "tac.h"
#include "block.hpp"

std::string operation_text(OPERATION op);
std::string operator_text(OPERATOR op);
std::string layout_text(LAYOUT layout);
std::string tac_text(tac_t& tac);
std::string tac_typesig_text(tac_t& tac, operand_t* scope);
std::string tac_layout_text(tac_t& tac, operand_t* scope);

int tac_noperands(tac_t& tac);
bool is_contiguous(operand_t& arg);

/* these should be part of core */
void bh_string_option(char *&option, const char *env_name, const char *conf_name);
void bh_path_option(char *&option, const char *env_name, const char *conf_name);
int error(int errnum, const char *fmt, ...);
int error(const char *err_msg, const char *fmt, ...);

#endif