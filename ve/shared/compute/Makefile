ROOT=../../..
include $(ROOT)/platform.mk
CXX=g++
#CXX=clang
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -std=c++98 -fPIC -I$(ROOT)/include
LDFLAGS= $(ARCH_OPTS) -L$(ROOT)/core -lbh -lrt -shared -Wl,$(LD_LIB_COMMAND)
HEADER=./*.hpp $(ROOT)/include/*
SRC= $(ALL_CPP_FILES) $(ALL_C_FILES)
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)

all: $(OBJ)

%.o: %.c $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

%.o: %.cpp $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

install: all
	@-(echo 'Done.')

clean:
	rm -rf *.o *~
