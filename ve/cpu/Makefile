ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_ve_cpu.$(LIB_SUFFIX)
CXX?=g++
#CXX?=clang
WARNOPTS=-Wall
EXTRAS+=
ifeq ($(PLATFORM_OS),OSX)
	EXTRAS+=-I$(ROOT)/thirdparty/ctemplate/include -I$(ROOT)/thirdparty/jansson/include
	EXTRA_LIBS+=-L$(ROOT)/thirdparty/jansson/osx-lib -L$(ROOT)/thirdparty/ctemplate/osx-lib 
endif

LCFLAGS=$(WARNOPTS) $(ARCH_OPTS) -std=c++0x -g -O2 -fPIC -I$(ROOT)/include -I$(ROOT)/core -I$(ROOT)/traverse $(EXTRAS)
LDFLAGS=-L../../core $(ARCH_OPTS) -lbh $(EXTRA_LIBS) -lctemplate -shared -Wl,$(LD_LIB_COMMAND)
HEADER=./*.h $(ROOT)/include/*
SRC= $(ALL_CPP_FILES) $(ALL_C_FILES)
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o) 
OBJ=bh_ve_cpu.o 

INSTALLDIR?=/opt/bohrium/

all: $(LIB)

%.o: %.c $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)  -I$(ROOT)/include

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include

precompile:
		/usr/bin/time ./precompiler

clean:
		rm -rf *.o *~ *.$(LIB_SUFFIX) kernels/* objects/* core precompiler
		touch kernels/empty
		touch objects/empty

install: all
		cp $(LIB) $(INSTALLDIR)/
		mkdir -p $(INSTALLDIR)/cpu/kernels
		mkdir -p $(INSTALLDIR)/cpu/objects
		mkdir -p $(INSTALLDIR)/cpu/templates
		cp templates/*.tpl $(INSTALLDIR)/cpu/templates/
		cp -r include $(INSTALLDIR)/cpu/include
