#ifndef __BH_FILTER_COMPOSITE_EXPANDER
#define __BH_FILTER_COMPOSITE_EXPANDER

#include <bh_component.hpp>

namespace bohrium {
namespace filter {
namespace bcexp {

extern bool __verbose;
extern void verbose_print(std::string str);

template <typename T>
inline void bh_set_constant(bh_instruction& instr, int opr_idx, bh_type type, T value);

class Expander
{
public:
    /**
     *  Construct the expander.
     */
    Expander(bool verbose, size_t threshold, int sign, int powk, int reduce_1d, int repeat);

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
     *  Make sure that BH_FREE has been sent down the stack.
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
    bh_view make_temp(bh_view& meta, bh_type type, bh_index nelem);
    bh_view make_temp(bh_type type, bh_index nelem);

    /**
     *  Inject an instruction.
     */

    // System instruction
    void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out);
    inline void inject(bh_ir& bhir, int pc, bh_instruction instr);

    // Unary instruction
    void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1);

    // Unary with constants
    template <typename T>
    inline void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, T in1, bh_type const_type);
    template <typename T>
    inline void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, T in1);

    // Binary instruction
    inline void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1, bh_view& in2);
    // Binary with constants
    template <typename T>
    inline void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1, T in2, bh_type const_type);

    template <typename T>
    inline void inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1, T in2);

    int expand_sign(bh_ir& bhir, int pc);
    int expand_powk(bh_ir& bhir, int pc);
    int expand_reduce1d(bh_ir& bhir, int pc, int fold_limit);
    int expand_repeat(bh_ir& bhir, int pc);

private:
    static const char TAG[];
    std::vector<bh_base*> bases_;
    size_t gc_threshold_;
    int sign_;
    int powk_;
    int reduce1d_;
    int repeat_;
};

void Expander::inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1, bh_view& in2)
{
    bh_instruction instr;
    instr.opcode = opcode;
    instr.operand[0] = out;
    instr.operand[1] = in1;
    instr.operand[2] = in2;
    bhir.instr_list.insert(bhir.instr_list.begin()+pc, instr);
}

inline void Expander::inject(bh_ir& bhir, int pc, bh_instruction instr)
{
    bhir.instr_list.insert(bhir.instr_list.begin()+pc, instr);
}

template <typename T>
inline void Expander::inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1, T in2, bh_type const_type)
{
    bh_instruction instr;
    instr.opcode = opcode;
    instr.operand[0] = out;
    instr.operand[1] = in1;
    bh_set_constant(instr, 2, const_type, in2);
    bhir.instr_list.insert(bhir.instr_list.begin()+pc, instr);
}

template <typename T>
inline void Expander::inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, bh_view& in1, T in2)
{
    Expander::inject(bhir, pc, opcode, out, in1, in2, in1.base->type);
}

template <typename T>
inline void Expander::inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, T in1, bh_type const_type)
{
    bh_instruction instr;
    instr.opcode = opcode;
    instr.operand[0] = out;
    bh_set_constant(instr, 1, const_type, in1);
    bhir.instr_list.insert(bhir.instr_list.begin()+pc, instr);
}

template <typename T>
inline void Expander::inject(bh_ir& bhir, int pc, bh_opcode opcode, bh_view& out, T in1)
{
    Expander::inject(bhir, pc, opcode, out, in1, out.base->type);
}

template <typename T>
inline void bh_set_constant(bh_instruction& instr, int opr_idx, bh_type type, T value)
{
    bh_flag_constant(&instr.operand[opr_idx]);
    instr.constant.type = type;

    switch(type) {
        case BH_BOOL:
            instr.constant.value.bool8 = (bh_bool)value;
            break;
        case BH_INT8:
            instr.constant.value.int8 = (bh_int8)value;
            break;
        case BH_INT16:
            instr.constant.value.int16 = (bh_int16)value;
            break;
        case BH_INT32:
            instr.constant.value.int32 = (bh_int32)value;
            break;
        case BH_INT64:
            instr.constant.value.int64 = (bh_int64)value;
            break;
        case BH_UINT8:
            instr.constant.value.uint8 = (bh_uint8)value;
            break;
        case BH_UINT16:
            instr.constant.value.uint16 = (bh_uint16)value;
            break;
        case BH_UINT32:
            instr.constant.value.uint32 = (bh_uint32)value;
            break;
        case BH_UINT64:
            instr.constant.value.uint64 = (bh_uint64)value;
            break;
        case BH_FLOAT32:
            instr.constant.value.float32 = (bh_float32)value;
            break;
        case BH_FLOAT64:
            instr.constant.value.float64 = (bh_float64)value;
            break;
        case BH_COMPLEX64:
            instr.constant.value.complex64.real = (float)value;
            instr.constant.value.complex64.imag = (float)0.0;
            break;
        case BH_COMPLEX128:
            instr.constant.value.complex128.real = (double)value;
            instr.constant.value.complex128.imag = (double)0.0;
            break;
        case BH_R123:
        case BH_UNKNOWN:
        default:
            fprintf(stderr, "set_constant unsupported for given type.");
            break;
    }
}

}}}
#endif
