<!--(for op, opcode, optype, opcount, _typesigs, _layouts, _broadcast in data)-->

// @!op!@ - @!opcode!@ - @!optype!@ - @!opcount!@ (A)
template <typename T>
inline
void @!op!@ (multi_array<T>& res)
{
    Runtime::instance().typecheck<@!opcode!@, T>();
    Runtime::instance().enqueue((bh_opcode)@!opcode!@, res);
}
<!--(end)-->
