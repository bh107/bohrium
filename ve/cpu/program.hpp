#ifndef __KP_PROGRAM_HPP
#define __KP_PROGRAM_HPP 1
#include <string>
#include "bh.h"
#include "kp.h"

namespace kp{
namespace core{

class Program {
public:
    /**
     * Construct a CAPE program, that is a topologically sorted
     * array of instructions and a SymbolTable.
     */
    Program(int64_t length);

    /**
     * Deconstructor
     */
    ~Program(void);

    /**
     * Amount of tacs that can be stored in the program.
     */
    size_t capacity(void);

    /**
     * Amount of tacs current stored in the program.
     */
    size_t size(void);

    void clear(void);

    kp_tac& operator[](size_t tac_idx);
    kp_tac* tacs(void);

    std::string text_meta(void);

private:
    Program(void);  // We do not wish to construct programs of unknown length.
    kp_program program_;

    static const char TAG[];
};

}}

#endif
