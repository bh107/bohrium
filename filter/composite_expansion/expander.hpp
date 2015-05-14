#ifndef __BH_FILTER_COMPOSITE_EXPANDER
#define __BH_FILTER_COMPOSITE_EXPANDER
#include "bh.h"

namespace bohrium {
namespace filter {
namespace composite {

void bh_set_constant(bh_instruction& instr, int opr_idx, bh_type type, float value);

class Expander
{
public:
    /**
     *  Construct the expander.
     */
    Expander(size_t threshold, int matmul, int sign);

    /**
     *  Tear down the expander.
     *  This includes freeing allocated bh_base pointers.
     */
    ~Expander(void);

    /**
     *  Modifies the given bhir, expanding composites per configuration.
     */
    void expand(bh_ir& bhir);

    /**
     *  Collect garbage, that is de-allocate an amount of bh_base.
     *
     *  Make sure that BH_FREE and BH_DISCARD has been sent down the stack.
     *
     *  It seems likely that re-use would be an idea so default
     *  strategy should probably be to keep some amount for re-use
     *  and de-allocate above some threshold.
     *
     *  Upon deconstruction everything should of course be de-allocated.
     */
    size_t gc(void);

    /**
     * Create a bh_base and return a pointer to it.
     *
     * @return Pointer to the created base.
     */
    bh_base* make_base(bh_type type, bh_index nelem);

    /**
     *  Inject an instruction.
     */
    void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1, bh_view& in2);
    void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1, double in2);
    void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1);
    void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, double);
    void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out);

    /**
     *  Expand matmul at the given pc in the bhir instruction list.
     *
     *  Returns the number of additional instructions used.
     */
    int expand_matmul(bh_ir& bhir, int pc);

    /**
     *  Expand BH_SIGN at the given PC into the sequence:
     *
     *      BH_SIGN OUT, IN1 (When IN1.type != COMPLEX):
     *
     *      LESS T1_BOOL, IN1, 0
     *      IDENTITY T1, T1_BOOL
     *      FREE T1_BOOL
     *      DISCARD T1_BOOL
     * 
     *      GREATER T2_BOOL, IN1, 0
     *      IDENTITY T2, T2_BOOL
     *      FREE T2_BOOL
     *      DISCARD T2_BOOL
     *
     *      SUBTRACT OUT, T2, T1
     *      FREE T1
     *      DISCARD T1
     *      FREE T2
     *      DISCARD T2
     *
     *                          ||| OR |||
     *
     *      BH_SIGN OUT, IN1 (When IN1.type == COMPLEX):
     *
     *      REAL IN1_R, IN1
     *
     *      LESS T1_BOOL, IN1_R, 0
     *      IDENTITY T1, T1_BOOL
     *      FREE T1_BOOL
     *      DISCARD T1_BOOL
     * 
     *      GREATER T2_BOOL, IN1_R, 0
     *      IDENTITY T2, T2_BOOL
     *      FREE T2_BOOL
     *      DISCARD T2_BOOL
     *
     *      FREE IN1_R
     *      DISCARD IN1_R
     *
     *      SUBTRACT T3, T2, T1
     *      IDENTITY OUT, T3
     *      FREE T1
     *      DISCARD T1
     *      FREE T2
     *      DISCARD T2
     *
     *  Returns the number of instructions used (12 or 17).
     */
    int expand_sign(bh_ir& bhir, int pc);

private:

    static const char TAG[];
    std::vector<bh_base*> bases_;
    size_t gc_threshold_;
    int matmul_;
    int sign_;
    
};

}}}
#endif
