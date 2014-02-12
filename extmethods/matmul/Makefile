ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_matmul.$(LIB_SUFFIX)
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -I$(ROOT)/include
LDFLAGS= $(ARCH_OPTS) -L$(ROOT)/core -lbh $(EXTRA_LIBS) -shared -lm -Wl,$(LD_LIB_COMMAND)
HEADER=./*.h $(ROOT)/include/*
SRC= $(ALL_CPP_FILES) $(ALL_C_FILES)
OBJ=$(SRC:.cpp=.o)
INSTALLDIR?=/opt/bh

all: $(LIB)

%.o: %.cpp $(HEADER) Makefile
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS) -I$(ROOT)/include

$(LIB): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include

clean:
	rm -rf *.o *~ *.$(LIB_SUFFIX)

install: all
	cp $(LIB) $(INSTALLDIR)
