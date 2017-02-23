case ${utype}: {
    ${alpha}
    ${beta}
    cblas_${t}${name}(
        ${if_layout} CblasRowMajor, ${endif_layout}
        ${if_side}   CblasLeft,     ${endif_side}
        ${if_uplo}   CblasUpper,    ${endif_uplo}
        ${if_transA} CblasNoTrans,  ${endif_transA}
        ${if_transB} CblasNoTrans,  ${endif_transB}
        ${if_diag}   CblasUnit,     ${endif_diag}
        ${if_m}      m,             ${endif_m}
        ${if_n}      n,             ${endif_n}
        ${if_k}      k,             ${endif_k}
        ${alpha_arg},
        (${type}*) A_data,
        k,
        ${if_B}
        (${type}*) B_data,
        n${if_C},${endif_C}
        ${endif_B}
        ${if_C}
        ${beta_arg},
        (${type}*) C_data,
        n
        ${endif_C}
    );
    break;
}
