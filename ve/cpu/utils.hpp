#ifndef __BH_VE_CPU_UTILS
#define __BH_VE_CPU_UTILS
#include "bh.h"
#include "tac.h"
#include "block.hpp"

#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdarg>
#include <memory>
#include <cerrno>
#include <fcntl.h>

#ifdef DEBUGGING
#define DEBUG(tag,x) do { std::cerr << TAG << "::" << x << std::endl; } while (0);
#else
#define DEBUG(tag,x)
#endif

namespace bohrium {
namespace utils {

//
// Self-explanatory function returning the textual representation of 
// TAC enums; useful for pretty-printing.
//
std::string operation_text(OPERATION op);
std::string operator_text(OPERATOR op);
std::string operand_text(const operand_t& operand);

std::string etype_text(ETYPE etype);
std::string etype_text_shand(ETYPE etype);
std::string etype_to_ctype_text(ETYPE etype);

std::string layout_text(LAYOUT layout);
std::string layout_text_shand(LAYOUT layout);

/**
 * Maps bh_type to ETYPE
 *
 * @param bhtype The bh_type to map.
 * @returns The ETYPE corresponding to the given bh_type.
 */
ETYPE bhtype_to_etype(bh_type bhtype);

/**
 * Returns a textual representation of a tac.
 */
std::string tac_text(const tac_t& tac);

int tac_noperands(const tac_t& tac);

/**
 *  Determine whether an operand has a contiguous layout.
 *
 *  @param arg The operand to inspect.
 *  @returns True when the layout is contiguous, false othervise.
 */
bool contiguous(const operand_t& arg);

/**
 *  Determines whether two operand have compatible meta-data.
 *
 *  This function serves the same purpose as bh_view_identical, 
 *  but for tac-operands instead of bh_instruction.operand[...].
 *
 *  @param one
 *  @param other
 *  @returns True when compatible false othervise.
 */
bool compatible(const operand_t& one, const operand_t& other);

/**
 *  Determines whether two operand have equivalent meta-data.
 *
 *  This function serves the same purpose as bh_view_identical, 
 *  but for tac-operands instead of bh_instruction.operand[...].
 *
 */
bool equivalent(const operand_t& one, const operand_t& other);

/**
 *  Return a string formatted with "fmt_str"; supporting positional identifiers.
 */
std::string string_format(const std::string fmt_str, ...);

/**
 *  Write source-code to file.
 *  Filename will be along the lines of: kernel/<symbol>_<UID>.c
 *  NOTE: Does not overwrite existing files.
 */
bool write_file(std::string file_path, const char* sourcecode, size_t source_len);

/**
 *  Returns a hash represented as a positive integer of the given text.
 *
 *  @param text The text to hash. 
 *  @returns Hash of the given text as an integer.
 */
uint32_t hash(std::string text);

/**
 * Returns a hash represented as a string of the given text.
 *
 * @param text The text to hash.
 * @returns Hash of the given text as a string.
 */ 
std::string hash_text(std::string text);

/* these should be part of core */
void bh_string_option(char *&option, const char *env_name, const char *conf_name);
void bh_path_option(char *&option, const char *env_name, const char *conf_name);

int error(int errnum, const char *fmt, ...);
int error(const char *err_msg, const char *fmt, ...);

}}
#endif
