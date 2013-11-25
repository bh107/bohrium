ROOT=../..
LIB=libbh_filter_papi.so
CXX?=g++
WARNOPTS=-Wall
EXTRAS+=
TIMING+=
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include -I$(ROOT)/core $(EXTRAS)
LDFLAGS=-L../../core -lbh -lrt -shared -Wl,-soname
HEADER=./*.h $(ROOT)/include/*
SRC= $(shell find -iname '*.cpp') $(shell find -iname '*.c')
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)
INSTALLDIR?=/opt/bohrium/

all: $(LIB)

%.o: %.cpp $(HEADER)
	$(CXX) -c $< -o $@  $(LCFLAGS) $(CFLAGS) -I$(ROOT)/include

$(LIB): $(OBJ)
	$(CXX) $(OBJ) -o $@ -L/usr/local/lib $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include /usr/local/lib/libpapi.a

clean:
	rm -rf *.o *~ *.so

install: all
	cp $(LIB) $(INSTALLDIR)

