struct @!uname!@Impl : public ExtmethodImpl {
public:
    void execute(bh_instruction *instr, void* arg) {
        // All matrices must be contigous
        assert(instr->isContiguous());

        // B is a n-by-nrhs matrix
        bh_view* B = &instr->operand[2];
        // We allocate the B data, if not already present
        bh_data_malloc(B->base);

        void *B_data = B->base->data;

        int n    = B->shape[0];
        int nrhs = B->ndim == 1 ? 1 : B->shape[1];
        int ldb  = n;

        <!--(if if_A)-->
            // A is a n-by-n square matrix
            bh_view* A = &instr->operand[1];
            // We allocate the A data, if not already present
            bh_data_malloc(A->base);
            // Grab pointer to data
            void *A_data = A->base->data;

            int lda = n;

            assert(A->base->type == B->base->type);
        <!--(end)-->

        <!--(if if_AB)-->
            bh_view* AB = &instr->operand[1];
            bh_data_malloc(AB->base);
            void *AB_data = AB->base->data;

            <!--(if if_klku)-->
                // kl is the number of non-zero elements in the first column of A minus 1
                int kl = 0;
                // ku is the number of non-zero elements in the first row of A minus 1
                int ku = 0;

                switch(AB->base->type) {
                    case bh_type::FLOAT32: {
                        kl = calc_kl<bh_float32>((bh_float32*) AB_data, AB->shape[0], AB->shape[1]);
                        ku = calc_ku<bh_float32>((bh_float32*) AB_data, AB->shape[0]);
                        break;
                    }
                    case bh_type::FLOAT64: {
                        kl = calc_kl<bh_float64>((bh_float64*) AB_data, AB->shape[0], AB->shape[1]);
                        ku = calc_ku<bh_float64>((bh_float64*) AB_data, AB->shape[0]);
                        break;
                    }
                    case bh_type::COMPLEX64: {
                        throw std::runtime_error("Not implemented yet!");
                    }
                    case bh_type::COMPLEX128: {
                        throw std::runtime_error("Not implemented yet!");
                    }
                    default:
                        std::stringstream ss;
                        ss << bh_type_text(AB->base->type) << " not supported by LAPACK for '@!name!@'.";
                        throw std::runtime_error(ss.str());
                }

                // ldab is the second dimension of the banded matrix
                int lda = 2 * kl + ku + 1;

                switch(AB->base->type) {
                    case bh_type::FLOAT32: {
                        AB_data = get_ab_data<bh_float32>((bh_float32*) AB_data, AB->shape[0], AB->shape[1], kl, ku);
                        break;
                    }
                    case bh_type::FLOAT64: {
                        AB_data = get_ab_data<bh_float64>((bh_float64*) AB_data, AB->shape[0], AB->shape[1], kl, ku);
                        break;
                    }
                    case bh_type::COMPLEX64: {
                        throw std::runtime_error("Not implemented yet!");
                    }
                    case bh_type::COMPLEX128: {
                        throw std::runtime_error("Not implemented yet!");
                    }
                    default:
                        std::stringstream ss;
                        ss << bh_type_text(AB->base->type) << " not supported by LAPACK for '@!name!@'.";
                        throw std::runtime_error(ss.str());
                }
            <!--(end)-->
        <!--(end)-->

        <!--(if if_DLDDU)-->
            // A is a n-by-n square matrix
            bh_view* A = &instr->operand[1];
            // We allocate the A data, if not already present
            bh_data_malloc(A->base);
            // Grab pointer to data
            void *A_data = A->base->data;
            assert(A->base->type == B->base->type);

            // DL is the elements of the n-1 sub-diagonal of A
            // D is the elements of the diagonal of A
            // DU is the elements of the n-1 super-diagonal of A
            void *DL, *D, *DU;

            switch(A->base->type) {
                case bh_type::FLOAT32: {
                    DL = get_subdiagonal<bh_float32>(A_data, n);
                    D  = get_diagonal<bh_float32>(A_data, n);
                    DU = get_superdiagonal<bh_float32>(A_data, n);
                    break;
                }
                case bh_type::FLOAT64: {
                    DL = get_subdiagonal<bh_float64>(A_data, n);
                    D  = get_diagonal<bh_float64>(A_data, n);
                    DU = get_superdiagonal<bh_float64>(A_data, n);
                    break;
                }
                case bh_type::COMPLEX64: {
                    throw std::runtime_error("Not implemented yet!");
                }
                case bh_type::COMPLEX128: {
                    throw std::runtime_error("Not implemented yet!");
                }
                default:
                    std::stringstream ss;
                    ss << bh_type_text(A->base->type) << " not supported by LAPACK for '@!name!@'.";
                    throw std::runtime_error(ss.str());
            }
        <!--(end)-->

        <!--(if if_AP)-->
            // AP is a the upper triangluar part of a matrix A in packed storage
            // Its dimensions is at least max(1, n(n+1)/2)
            bh_view* AP = &instr->operand[1];
            // We allocate the AP data, if not already present
            bh_data_malloc(AP->base);
            // Grab pointer to data
            void *AP_data = AP->base->data;

            assert(AP->base->type == B->base->type);

            // Convert the matrix into an array in upper-packed storage mode.
            // https://www.ibm.com/support/knowledgecenter/SSFHY8_5.3.0/com.ibm.cluster.essl.v5r3.essl100.doc/am5gr_upsm.htm

            switch(B->base->type) {
                case bh_type::FLOAT32: {
                    AP_data = get_ap_data<bh_float32>((bh_float32*) AP_data, n);
                    break;
                }
                case bh_type::FLOAT64: {
                    AP_data = get_ap_data<bh_float64>((bh_float64*) AP_data, n);
                    break;
                }
                case bh_type::COMPLEX64: {
                    throw std::runtime_error("Not implemented yet!");
                }
                case bh_type::COMPLEX128: {
                    throw std::runtime_error("Not implemented yet!");
                }
                default:
                    std::stringstream ss;
                    ss << bh_type_text(B->base->type) << " not supported by LAPACK for '@!name!@'.";
                    throw std::runtime_error(ss.str());
            }
        <!--(end)-->

        <!--(if if_uplo)-->
            char uplo = 'U';
        <!--(end)-->

        <!--(if if_ipiv)-->
            int *ipiv = NULL;
            ipiv = new int[n];
        <!--(end)-->

        int info;

        switch(B->base->type) {
            @!func!@
            default:
                std::stringstream ss;
                ss << bh_type_text(B->base->type) << " not supported by LAPACK for '@!name!@'.";
                throw std::runtime_error(ss.str());
        } /* end of switch */
    } /* end execute method */
}; /* end of struct */
