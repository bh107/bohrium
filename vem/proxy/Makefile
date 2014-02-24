ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_proxy_server.$(LIB_SUFFIX)
SERV=bh_proxy_client
CC?=gcc
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -std=gnu++0x -I$(ROOT)/include -I$(ROOT)/iniparser/src
LDFLAGS= $(ARCH_OPTS) -shared -L../../core -lbh -Wl,$(LD_LIB_COMMAND),$(LIB)
SRC=$(ALL_CPP_FILES)
HEADER=$(ALL_H_FILES) $(ALL_HPP_FILES)
OBJ=$(SRC:.cpp=.o)
INSTALLDIR?=/opt/bohrium/

all: $(SERV) $(LIB)

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS)

$(SERV): $(OBJ)
		$(CXX) $(OBJ) -o $@ -Wall -I../../include -L../../core -lbh
clean:
		rm -rf *.o *~ $(LIB) $(SERV)

install: all
		cp $(LIB) $(INSTALLDIR)
		cp $(SERV) $(INSTALLDIR)
