case ${utype}: {
    ${alpha}
    ${beta}
    clblas${t}${name}(
        ${if_layout} clblasRowMajor, ${endif_layout}
        ${if_side}   clblasLeft,     ${endif_side}
        ${if_uplo}   clblasUpper,    ${endif_uplo}
        ${if_transA} clblasNoTrans,  ${endif_transA}
        ${if_transB} clblasNoTrans,  ${endif_transB}
        ${if_diag}   clblasUnit,     ${endif_diag}
        ${if_m}      m,              ${endif_m}
        ${if_n}      n,              ${endif_n}
        ${if_k}      k,              ${endif_k}
        ${alpha_arg},
        bufA,
        0,
        k,
        ${if_B}
        bufB,
        0,
        n,
        ${endif_B}
        ${if_C}
        ${beta_arg},
        bufC,
        0,
        n,
        ${endif_C}
        1,
        &queue,
        0,
        NULL,
        &event
    );
    break;
}
