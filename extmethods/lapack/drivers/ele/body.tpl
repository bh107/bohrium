struct @!uname!@Impl : public ExtmethodImpl {
public:
    void execute(bh_instruction *instr, void* arg) {
        throw std::runtime_error("@!name!@ not supported by LAPACK extmethod (yet!).");
    } /* end execute method */
}; /* end of struct */
