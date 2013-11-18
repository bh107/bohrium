PLATFORM_OS:=Unknown

ifeq ($(OS),Windows_NT)
# Windows
    PLATFORM_OS:=Windows
    INSTALLDIR=C:\

    XBUILD=msbuild 
    CCFLAGS += -D WIN32
    ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
        CCFLAGS += -D AMD64
    endif
    ifeq ($(PROCESSOR_ARCHITECTURE),x86)
        CCFLAGS += -D IA32
    endif
else
#Linux or OSX
    PLATFORM_OS:=Linux

    XBUILD?=xbuild
    INSTALLDIR?=/opt/bohrium/

    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)

        CCFLAGS += -D LINUX

        LIB_SUFFIX=so
        LD_LIB_COMMAND=-soname
        EXTRA_LIBS=-lrt
        ALL_C_FILES=$(shell find -iname '*.c')
        ALL_H_FILES=$(shell find -iname '*.h')
        ALL_CPP_FILES=$(shell find -iname '*.cpp')
        ALL_HPP_FILES=$(shell find -iname '*.hpp')

        UNAME_P := $(shell uname -p)
        ifeq ($(UNAME_P),x86_64)
            CCFLAGS += -D AMD64
        endif
        ifneq ($(filter %86,$(UNAME_P)),)
            CCFLAGS += -D IA32
        endif
        ifneq ($(filter arm%,$(UNAME_P)),)
            CCFLAGS += -D ARM
        endif        
    endif
    ifeq ($(UNAME_S),Darwin)

        PLATFORM_OS:=OSX
        CCFLAGS += -D OSX

        LIB_SUFFIX=dylib
        LD_LIB_COMMAND=-dylib_install_name
        EXTRA_LIBS=
        ARCH_OPTS=-arch i386 -arch x86_64
        ALL_C_FILES=$(shell find . -iname '*.c')
        ALL_H_FILES=$(shell find . -iname '*.h')
        ALL_CPP_FILES=$(shell find . -iname '*.cpp')
        ALL_HPP_FILES=$(shell find . -iname '*.hpp')
    endif
endif



