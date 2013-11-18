ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_vem_node.$(LIB_SUFFIX)
CC?=gcc
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -std=gnu++0x -I$(ROOT)/include -I$(ROOT)/iniparser/src
LDFLAGS= $(ARCH_OPTS) -shared -L../../core -lbh -Wl,$(LD_LIB_COMMAND),$(LIB)
SRC=$(ALL_CPP_FILES)
HEADER=$(ALL_H_FILES) $(ALL_HPP_FILES)
OBJ=$(SRC:.cpp=.o)
INSTALLDIR?=/opt/bohrium/

all: $(LIB)

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB)

install: all
		cp $(LIB) $(INSTALLDIR)
