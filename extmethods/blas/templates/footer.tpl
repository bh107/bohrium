
extern "C" ExtmethodImpl* blas_@!name!@_create() {
    return new @!uname!@Impl();
}

extern "C" void blas_@!name!@_destroy(ExtmethodImpl* self) {
    delete self;
}
