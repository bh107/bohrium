ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_filter_transitive_reduction.$(LIB_SUFFIX)
CXX?=g++
WARNOPTS=-Wall
EXTRAS+=
TIMING+=
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -I$(ROOT)/include -I$(ROOT)/core -I$(ROOT)/traverse $(EXTRAS)
LDFLAGS= $(ARCH_OPTS) -L$(ROOT)/core -lbh $(EXTRA_LIBS) -shared -Wl,$(LD_LIB_COMMAND)
HEADER=./*.h $(ROOT)/include/*
SRC=$(ALL_CPP_FILES) $(ALL_C_FILES)
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)
INSTALLDIR?=/opt/bohrium/

all: $(LIB)

%.o: %.c $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS) -I$(ROOT)/include

%.o: %.cpp $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS) -I$(ROOT)/include

$(LIB): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include

clean:
	rm -rf *.o *~ *.$(LIB_SUFFIX)

install: all
	cp $(LIB) $(INSTALLDIR)

