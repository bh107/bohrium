case @!utype!@: {
    @!alpha!@
    @!beta!@
    cblas_@!t!@@!name!@(
        <!--(if if_layout)-->   CblasRowMajor, <!--(end)-->
        <!--(if if_side)-->     CblasLeft,     <!--(end)-->
        <!--(if if_uplo)-->     CblasUpper,    <!--(end)-->
        <!--(if if_notransA)--> CblasNoTrans,  <!--(end)-->
        <!--(if if_transA)-->   CblasTrans,    <!--(end)-->
        <!--(if if_notransB)--> CblasNoTrans,  <!--(end)-->
        <!--(if if_diag)-->     CblasUnit,     <!--(end)-->
        <!--(if if_m)-->        m,             <!--(end)-->
        <!--(if if_n)-->        n,             <!--(end)-->
        <!--(if if_k)-->        k,             <!--(end)-->
        @!alpha_arg!@,
        (@!blas_type!@*)(((@!type!@*) A_data) + A->start),
        k,
        <!--(if if_B)-->
        (@!blas_type!@*)(((@!type!@*) B_data) + B->start),
        n<!--(if if_C)-->,<!--(end)-->
        <!--(end)-->
        <!--(if if_C)-->
        @!beta_arg!@,
        (@!blas_type!@*)(((@!type!@*) C_data) + C->start),
        n
        <!--(end)-->
    );
    break;
}
