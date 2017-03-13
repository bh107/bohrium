case @!utype!@: {
    LAPACK_FUN(@!t!@@!name!@)(
        <!--(if if_uplo)--> &uplo, <!--(end)-->

        &n,

        <!--(if if_klku)-->
            &kl,
            &ku,
        <!--(end)-->

        &nrhs,

        <!--(if if_A)-->    ((@!type!@*) A_data) + A->start,   <!--(end)-->
        <!--(if if_AB)-->   ((@!type!@*) AB_data) + AB->start, <!--(end)-->
        <!--(if if_AP)-->   ((@!type!@*) AP_data) + AP->start, <!--(end)-->
        <!--(if if_lda)-->  &lda,                              <!--(end)-->
        <!--(if if_ipiv)--> ipiv,                              <!--(end)-->

        <!--(if if_DLDDU)-->
            (@!type!@*) DL,
            (@!type!@*) D,
            (@!type!@*) DU,
        <!--(end)-->

        ((@!type!@*) B_data) + B->start,
        &ldb,

        &info
    );
    break;
}
