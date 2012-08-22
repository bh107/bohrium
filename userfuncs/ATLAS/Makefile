ROOT=../..
LIB=libcphvb_atlas.so
CXX?=g++
WARNOPTS=-Wall
ATLASDIR?=/usr/local/atlas
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include -I$(ATLASDIR)/include
LDFLAGS= -L$(ATLASDIR)/lib -ltatlas -fopenmp -L$(ROOT)/core -lcphvb -lrt -shared -Wl,-soname
HEADER=$(ROOT)/include/*  
SRC= $(shell find -iname '*.cpp') $(shell find -iname '*.c')
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)
INSTALLDIR?=/opt/cphvb/

all: $(LIB)

%.o: %.c $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)  -I$(ROOT)/include

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include

clean:
		rm -rf *.o *~ *.so

install: all
		cp $(LIB) $(INSTALLDIR)

