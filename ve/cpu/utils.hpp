#ifndef __KP_CORE_UTILS_HPP
#define __KP_CORE_UTILS_HPP 1
#include <fstream>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdarg>
#include <vector>
#include <memory>
#include <cerrno>
#include <fcntl.h>

#include <bh_pprint.h>
#include <bh_ir.hpp>

#include "kp.h"
#include "block.hpp"
#include "symbol_table.hpp"

#ifdef DEBUGGING
#define DEBUG(tag,x) do { std::cerr << TAG << "::" << x << std::endl; } while (0);
#else
#define DEBUG(tag,x)
#endif

namespace kp{
namespace core{

template <typename T>
std::string to_string(T val);

//
// Self-explanatory function returning the textual representation of
// TAC enums; useful for pretty-printing.
//

std::string iterspace_text(const kp_iterspace & iterspace);
std::string omask_text(uint32_t omask);
std::string omask_aop_text(uint32_t omask);
std::string operation_text(KP_OPERATION op);
std::string operator_text(KP_OPERATOR op);
std::string operand_text(const kp_operand & operand);
std::string operand_access_text(const kp_operand & operand);

std::string etype_text(KP_ETYPE etype);
std::string etype_text_shand(KP_ETYPE etype);
std::string etype_to_ctype_text(KP_ETYPE etype);

std::string layout_text(KP_LAYOUT layout);
std::string layout_text_shand(KP_LAYOUT layout);

/**
 * Maps bh_type to KP_ETYPE
 *
 * @param bhtype The bh_type to map.
 * @returns The KP_ETYPE corresponding to the given bh_type.
 */
        KP_ETYPE bhtype_to_etype(bh_type bhtype);

/**
 * Returns a textual representation of a tac.
 */
std::string tac_text(const kp_tac & tac);

/**
 * Returns a textual representation of a tac including kp_operand info.
 */
std::string tac_text(const kp_tac & tac, SymbolTable& symbol_table);

size_t tac_noperands(const kp_tac & tac);

/**
 *  Transforms the given tac to a KP_NOOP or an equivalent tac,
 *  which should be cheaper compute.
 *
 *  # Silly stuff like
 *
 *  KP_IDENTITY a, a   -> KP_NOOP
 *
 *  # Operators with scalar neutral element
 *
 *  ADD a, a, 0     -> KP_NOOP
 *  MUL b, b, 1     -> KP_NOOP
 *  DIV a, a, 1     -> KP_NOOP

 *  ADD a, b, 0     -> KP_IDENTITY a, b
 *  MUL a, b, 1     -> KP_IDENTITY a, b
 *  MUL a, b, 0     -> KP_IDENTITY a, 0
 *
 *  # Specialization
 *
 *  POW a, a, 2     -> MUL a, a, a
 */
void tac_transform(kp_tac & tac, SymbolTable& symbol_table);

/**
 *  Map bh_ir->instr_list (bh_instruction) to kp_tac with entries in symbol_table.
 */
void instrs_to_tacs(bh_ir& bhir,
                    Program& tacs,
                    SymbolTable& symbol_table);

/**
 *  Determine whether an kp_operand has a contiguous layout.
 *
 *  @param arg The kp_operand to inspect.
 *  @returns True when the layout is contiguous, false othervise.
 */
bool contiguous(const kp_operand & arg);

/**
 *  Determine KP_LAYOUT of the given kp_operand by inspecting the stride/shape.
 */
        KP_LAYOUT determine_layout(const kp_operand & arg);

/**
 *  Return the first element that arg.data points to.
 *
 *  NOTE: Type is converted but overflows are not handled.
 */
double get_scalar(const kp_operand & arg);

/**
 *  Set the first element that arg.data points to.
 *
 *  NOTE: Type is converted but overflows are not handled.
 */
void set_scalar(const kp_operand & arg, double value);

/**
 *  Determines whether two kp_operand have compatible meta-data.
 *
 *  This function serves the same purpose as bh_view_identical,
 *  but for tac-operands instead of bh_instruction.kp_operand[...].
 *
 *  @param one
 *  @param other
 *  @returns True when compatible false othervise.
 */
bool compatible(const kp_operand & one, const kp_operand & other);

/**
 *  Determines whether two kp_operand have equivalent meta-data.
 *
 *  This function serves the same purpose as bh_view_identical,
 *  but for tac-operands instead of bh_instruction.kp_operand[...].
 *
 */
bool equivalent(const kp_operand & one, const kp_operand & other);

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

/* these should be part of core
void bh_string_option(char *&option, const char *env_name, const char *conf_name);
void bh_path_option(char *&option, const char *env_name, const char *conf_name);
*/

int error(int errnum, const char *fmt, ...);
int error(const char *err_msg, const char *fmt, ...);

}}

#endif
