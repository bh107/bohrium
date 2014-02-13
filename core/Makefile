ROOT=..
include $(ROOT)/platform.mk
LIB=libbh.$(LIB_SUFFIX)
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -std=gnu++0x -I$(ROOT)/include
LDFLAGS= $(ARCH_OPTS) -shared -ldl $(EXTRA_LIBS) -pthread -Wl,$(LD_LIB_COMMAND),$(LIB)
SRC=$(ALL_CPP_FILES) $(ALL_C_FILES)
OBJC=$(SRC:.c=.o)
OBJ=$(OBJC:.cpp=.o) $(ROOT)/iniparser/libiniparser.a
HEADER=$(ALL_H_FILES) $(ALL_HPP_FILES) $(ROOT)/include/*



all: $(LIB)

%.o: %.cpp $(HEADER) Makefile
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

%.o: %.c $(HEADER)  Makefile
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf *.o *~ $(LIB)

install: all
	cp $(LIB) $(INSTALLDIR)
