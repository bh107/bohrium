case @!utype!@: {
    @!alpha!@
    @!beta!@
    clblas@!t!@@!name!@(
        <!--(if if_layout)-->   clblasRowMajor, <!--(end)-->
        <!--(if if_side)-->     clblasLeft,     <!--(end)-->
        <!--(if if_uplo)-->     clblasUpper,    <!--(end)-->
        <!--(if if_notransA)--> clblasNoTrans,  <!--(end)-->
        <!--(if if_transA)-->   clblasTrans,    <!--(end)-->
        <!--(if if_notransB)--> clblasNoTrans,  <!--(end)-->
        <!--(if if_diag)-->     clblasUnit,     <!--(end)-->
        <!--(if if_m)-->        m,              <!--(end)-->
        <!--(if if_n)-->        n,              <!--(end)-->
        <!--(if if_k)-->        k,              <!--(end)-->
        @!alpha_arg!@,
        bufA,
        0,
        k,
        <!--(if if_B)-->
        bufB,
        0,
        n,
        <!--(end)-->
        <!--(if if_C)-->
        @!beta_arg!@,
        bufC,
        0,
        n,
        <!--(end)-->
        1,
        &queue,
        0,
        NULL,
        &event
    );
    break;
}
