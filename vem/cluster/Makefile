ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_vem_cluster.$(LIB_SUFFIX)
SLAVE=bh_vem_cluster_slave
CXX=mpic++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -std=gnu++11 -fno-strict-aliasing -I$(ROOT)/include -I$(ROOT)/iniparser/src
LDFLAGS= $(ARCH_OPTS) -L../../core -lbh -ldl -o
HEADER=./*.h $(ROOT)/include/*
SRC= $(ALL_CPP_FILES) $(ALL_C_FILES)
OBJ=$(SRC:.cpp=.o)
INSTALLDIR?=/opt/bohrium/

all: $(LIB) $(SLAVE)

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ -shared $(LDFLAGS) $(LIB)

$(SLAVE): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS) $(SLAVE)

clean:
		rm -rf *.o *~ $(LIB) $(SLAVE)

install: all
		cp $(LIB) $(INSTALLDIR)
		cp $(SLAVE) $(INSTALLDIR)
