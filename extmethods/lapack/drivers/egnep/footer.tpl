
extern "C" ExtmethodImpl* lapack_@!name!@_create() {
    return new @!uname!@Impl();
}

extern "C" void lapack_@!name!@_destroy(ExtmethodImpl* self) {
    delete self;
}
