
extern "C" ExtmethodImpl* opencv_@!name!@_create() {
    return new @!uname!@Impl();
}

extern "C" void opencv_@!name!@_destroy(ExtmethodImpl* self) {
    delete self;
}
