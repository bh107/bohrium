<!--(for op, opcode, optype, opcount, _typesigs, layouts, _broadcast in data)-->
    <!--(if ["A", "A"] in layouts)-->
    // @!op!@ - @!opcode!@ - @!optype!@ - @!opcount!@ (A,A)
    template <typename OutT, typename InT>
    inline
    void @!op!@ (multi_array<OutT>& res, multi_array<InT> &rhs)
    {
        Runtime::instance().typecheck<@!opcode!@, OutT, InT>();
        Runtime::instance().enqueue(@!opcode!@, res, rhs);
    }
    <!--(end)-->

    <!--(if ["A", "K"] in layouts)-->
    // @!op!@ - @!opcode!@ - @!optype!@ - @!opcount!@ (A,K)
    template <typename OutT, typename InT>
    inline
    void @!op!@ (multi_array<OutT>& res, const InT rhs)
    {
        Runtime::instance().typecheck<@!opcode!@, OutT, InT>();
        Runtime::instance().enqueue(@!opcode!@, res, rhs);
    }
    <!--(end)-->
<!--(end)-->
