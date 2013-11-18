ROOT=../../..
include $(ROOT)/platform.mk
LIB=libbh_ve_tiling.$(LIB_SUFFIX)
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O3 -fPIC -I$(ROOT)/include  -I$(ROOT)/core -I$(ROOT)/traverse
LDFLAGS= $(ARCH_OPTS) -L$(ROOT)/core -lbh $(EXTRA_LIBS) -shared -Wl,$(LD_LIB_COMMAND)
HEADER=./*.h $(ROOT)/include/*
SRC= $(ALL_CPP_FILES) $(ALL_C_FILES)
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o) $(ROOT)/ve/shared/compute/bh_compute.o $(ROOT)/ve/shared/compute/bh_compute_helpers.o $(ROOT)/ve/shared/compute/bh_compute_random.o $(ROOT)/ve/shared/compute/bh_compute_reduce.o $(ROOT)/ve/shared/compute/bh_compute_matmul.o $(ROOT)/ve/shared/compute/bh_compute_nselect.o $(ROOT)/ve/shared/bundler/bh_bundler.o
INSTALLDIR?=/opt/bohrium/

all: $(LIB)

%.o: %.c $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)  -I$(ROOT)/include

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include

clean:
		rm -rf *.o *~ *.$(LIB_SUFFIX)

install: all
		cp $(LIB) $(INSTALLDIR)

