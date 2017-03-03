<!--(for op, opcode, _optype, _opcount, _typesigs, _layouts, _broadcast in data)-->
template <typename T>
inline
multi_array<T>& multi_array<T>::@!op!@ (const T rhs)
{
    @!opcode.lower()!@ (*this, rhs);
    return *this;
}

template <typename T>
inline
multi_array<T>& multi_array<T>::@!op!@ (multi_array<T>& rhs)
{
    @!opcode.lower()!@ (*this, rhs);
    return *this;
}
<!--(end)-->
