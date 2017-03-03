<!--(for op, opcode, _optype, _opcount, _typesigs, _layouts, _broadcast in data)-->
template <typename T>
inline
multi_array<T>& @!op!@ (multi_array<T>& rhs)
{
    multi_array<T>* res = &Runtime::instance().create_base<T, T>(rhs); // Construct result
    @!opcode.lower()!@ (*res, rhs); // Enqueue
    res->setTemp(true); // Mark result as temp

    return *res;
}
<!--(end)-->
