<!--(for op, opcode, optype, opcount, _typesigs, _layouts, _broadcast in data)-->
// @!op!@ - @!opcode!@ - @!optype!@ - @!opcount!@ (A,K,K)
template <typename T>
inline
void @!op!@ (multi_array<T>& res, uint64_t in1, uint64_t in2)
{
    Runtime::instance().typecheck<@!opcode!@, T, uint64_t, uint64_t>();
    Runtime::instance().enqueue((bh_opcode)@!opcode!@, res, in1, in2);
}
<!--(end)-->
