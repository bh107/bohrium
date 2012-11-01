ROOT=../..
LIB=libcphvb_vem_cluster.so
SLAVE=cphvb_vem_cluster_slave
CXX=mpic++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=gnu++0x -I$(ROOT)/include -I$(ROOT)/iniparser/src
LDFLAGS=-L../../core -lcphvb -Wl -o $(SLAVE)
HEADER=./*.h $(ROOT)/include/*
SRC= $(shell find -iname '*.cpp') $(shell find -iname '*.c')
OBJ=$(SRC:.cpp=.o)
INSTALLDIR?=/opt/cphvb/

all: $(LIB) $(SLAVE)

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ -shared $(LDFLAGS)

$(SLAVE): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS)

clean:
		rm -rf *.o *~ $(LIB) $(SLAVE)

install: all
		cp $(LIB) $(INSTALLDIR)
		cp $(SLAVE) $(INSTALLDIR)
