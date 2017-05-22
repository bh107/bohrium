<!--(for op, opcode, optype, opcount, _typesigs, _layouts, _broadcast in data)-->

// @!op!@ - @!opcode!@ - @!optype!@ - @!opcount!@ ()
inline
void @!op!@ (void)
{
    Runtime::instance().enqueue(@!opcode!@);
}
<!--(end)-->
