struct ${uname}Impl : public ExtmethodImpl {
private:
    cl_event event = NULL;
public:
    ${uname}Impl(void) {
        clblasSetup();
    };

    ~${uname}Impl(void) {
        clWaitForEvents(1, &event);
        clReleaseEvent(event);
        clblasTeardown();
    };

    void execute(bh_instruction *instr, void* arg) {
        // Get the engine from the argument
        EngineOpenCL* engine = (EngineOpenCL*) arg;
        cl_command_queue queue = engine->getCQueue();

        // All matrices must be contigous
        assert(instr->is_contiguous());

        // A is a m*k matrix
        bh_view* A = &instr->operand[1];
        cl_mem bufA = engine->getCBuffer(A->base);

        ${if_B}
        // B is a k*n matrix
        bh_view* B = &instr->operand[2];
        assert(A->base->type == B->base->type);
        cl_mem bufB = engine->getCBuffer(B->base);
        ${endif_B}

        ${if_C}
        // C is a m*n matrix
        bh_view* C = &instr->operand[0];

        // We allocate the C data, if not already present
        if (bh_data_malloc(C->base) != BH_SUCCESS) {
            cerr << "Cannot allocate memory for C-matrix" << endl;
            return;
        }

        assert(A->base->type == C->base->type);
        cl_mem bufC = engine->getCBuffer(C->base);
        ${endif_C}

        int k = A->shape[1];
        ${if_m} int m = A->shape[0]; ${endif_m}
        ${if_n} int n = ${second_matrix}->shape[1]; ${endif_n}

        // Make sure that everything is copied to device, before executing clBlas method
        clFinish(queue);

        switch(A->base->type) {
            ${func}
            default:
                std::stringstream ss;
                ss << bh_type_text(A->base->type) << " not supported by BLAS for '${name}'.";
                throw std::runtime_error(ss.str());
        } /* end of switch */
    }; /* end execute method */
}; /* end of struct */
