<!--(for op, opcode, optype, opcount, _typesigs, layouts, _broadcast in data)-->
    <!--(if ["A", "A", "A", "A"] in layouts)-->
    // @!op!@ - @!opcode!@ - @!optype!@ - @!opcount!@ (A,A,A,A)
    template <typename TO, typename T1, typename T2, typename T3>
    inline
    void @!op!@ (multi_array<TO>& res, multi_array<T1>& in1, multi_array<T2>& in2, multi_array<T3>& in3)
    {
        Runtime::instance().typecheck<@!opcode!@, TO, T1, T2, T3>();
        Runtime::instance().enqueue(@!opcode!@, res, in1, in2, in3);
    }
    <!--(end)-->

    <!--(if ["A", "A", "A"] in layouts)-->
    // @!op!@ - @!opcode!@ - @!optype!@ - @!opcount!@ (A,A,A)
    template <typename TO, typename TL, typename TR>
    inline
    void @!op!@ (multi_array<TO>& res, multi_array<TL>& lhs, multi_array<TR>& rhs)
    {
        Runtime::instance().typecheck<@!opcode!@, TO, TL, TR>();
        Runtime::instance().enqueue(@!opcode!@, res, lhs, rhs);
    }
    <!--(end)-->

    <!--(if ["A", "A", "K"] in layouts)-->
    // @!op!@ - @!opcode!@ - @!optype!@ - @!opcount!@ (A,A,K)
    template <typename TO, typename TL, typename TR>
    inline
    void @!op!@ (multi_array<TO>& res, multi_array<TL>& lhs, const TR rhs)
    {
        Runtime::instance().typecheck<@!opcode!@, TO, TL, TR>();
        Runtime::instance().enqueue(@!opcode!@, res, lhs, rhs);
    }
    <!--(end)-->

    <!--(if ["A", "K", "A"] in layouts)-->
    // @!op!@ - @!opcode!@ - @!optype!@ - @!opcount!@ (A,K,A)
    template <typename TO, typename TL, typename TR>
    inline
    void @!op!@ (multi_array<TO>& res, const TL lhs, multi_array<TR>& rhs)
    {
        Runtime::instance().typecheck<@!opcode!@, TO, TL, TR>();
        Runtime::instance().enqueue(@!opcode!@, res, lhs, rhs);
    }
    <!--(end)-->
<!--(end)-->
