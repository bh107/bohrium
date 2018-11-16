struct @!uname!@Impl : public ExtmethodImpl {
private:
    cl_event event = NULL;
public:
    @!uname!@Impl(void) {
        clblasSetup();
    };

    ~@!uname!@Impl(void) {
        clWaitForEvents(1, &event);
        clblasTeardown();
    };

    void execute(bh_instruction *instr, void* arg) {
        // Get the engine from the argument
        EngineOpenCL* engine = (EngineOpenCL*) arg;
        cl_command_queue queue = engine->getCQueue();

        // All matrices must be contigous
        assert(instr->isContiguous());

        // A is a m*k matrix
        bh_view* A = &instr->operand[1];
        cl_mem bufA = engine->getCBuffer(A->base);

        <!--(if if_B)-->
        // B is a k*n matrix
        bh_view* B = &instr->operand[2];
        assert(A->base->dtype() == B->base->dtype());
        cl_mem bufB = engine->getCBuffer(B->base);
        <!--(end)-->

        <!--(if if_C)-->
        // C is a m*n matrix
        bh_view* C = &instr->operand[0];

        // We allocate the C data, if not already present
        bh_data_malloc(C->base);

        assert(A->base->dtype() == C->base->dtype());
        cl_mem bufC = engine->getCBuffer(C->base);
        <!--(end)-->

        int k = A->shape[1];
        <!--(if if_m)-->
        int m = A->shape[0];
        <!--(end)-->
        <!--(if if_n)-->
            int n;
            <!--(if if_C)--> n = C->shape[1]; <!--(end)-->
            <!--(if if_B)--> n = B->shape[1]; <!--(end)-->
        <!--(end)-->

        // Make sure that everything is copied to device, before executing clBlas method
        clFinish(queue);

        switch(A->base->dtype()) {
            @!func!@
            default:
                std::stringstream ss;
                ss << bh_type_text(A->base->dtype()) << " not supported by clBLAS for '@!name!@'.";
                throw std::runtime_error(ss.str());
        } /* end of switch */
    }; /* end execute method */
}; /* end of struct */
