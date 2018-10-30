struct @!uname!@Impl : public ExtmethodImpl {
public:
    void execute(bh_instruction *instr, void* arg) {
        // All matrices must be contigous
        assert(instr->isContiguous());

        // A is a m*k matrix
        bh_view* A = &instr->operand[1];
        // We allocate the A data, if not already present
        bh_data_malloc(A->base);
        void *A_data = A->base->data;

        <!--(if if_B)-->
        // B is a k*n matrix
        bh_view* B = &instr->operand[2];
        // We allocate the B data, if not already present
        bh_data_malloc(B->base);

        assert(A->base->dtype() == B->base->dtype());
        void *B_data = B->base->data;
        <!--(end)-->

        <!--(if if_C)-->
        // C is a m*n matrix
        bh_view* C = &instr->operand[0];

        // We allocate the C data, if not already present
        bh_data_malloc(C->base);

        assert(A->base->dtype() == C->base->dtype());
        void *C_data = C->base->data;
        <!--(end)-->

        int k = A->shape[1];
        <!--(if if_m)--> int m = A->shape[0]; <!--(end)-->
        <!--(if if_n)-->
            int n;
            <!--(if if_C)--> n = C->shape[1]; <!--(end)-->
            <!--(if if_B)--> n = B->shape[1]; <!--(end)-->
        <!--(end)-->

        switch(A->base->dtype()) {
            @!func!@
            default:
                std::stringstream ss;
                ss << bh_type_text(A->base->dtype()) << " not supported by BLAS for '@!name!@'.";
                throw std::runtime_error(ss.str());
        } /* end of switch */
    } /* end execute method */
}; /* end of struct */
