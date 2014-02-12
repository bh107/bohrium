ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_fftw.$(LIB_SUFFIX)
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -I$(ROOT)/include
LDFLAGS= $(ARCH_OPTS) -L$(ROOT)/core -lbh -lfftw3_omp -lfftw3 -lfftw3f_omp -lfftw3f -lm -fopenmp $(EXTRA_LIBS) -shared -lm -Wl,$(LD_LIB_COMMAND)
HEADER=./*.h $(ROOT)/include/*
SRC= $(ALL_CPP_FILES) $(ALL_C_FILES)
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)
INSTALLDIR?=/opt/bh

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

