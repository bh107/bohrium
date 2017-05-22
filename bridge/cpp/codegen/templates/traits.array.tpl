template <typename T>
inline
void assign_array_type(bh_base* base) {
    // TODO: The general case should result in a meaningful compile-time error.
    std::cout << "Unsupported type: " << base << std::endl;
}

<!--(for ctype, _bh_atype, _bh_ctype, bh_enum in data)-->
template <>
inline
void assign_array_type<@!ctype!@>(bh_base* base)
{
    base->type = @!bh_enum!@;
}

<!--(end)-->

template <>
inline
void assign_array_type<bh_complex64>(bh_base* base)
{
    base->type = BH_COMPLEX64;
}

template <>
inline
void assign_array_type<bh_complex128>(bh_base* base)
{
    base->type = BH_COMPLEX128;
}
