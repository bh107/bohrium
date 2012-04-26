ROOT=../..
LIB=libcphvb_ve_simple.so
CXX?=g++
WARNOPTS=-Wall -pedantic
CFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include
CXXFLAGS= $(WARNOPTS) -g -O2 -fPIC -std=c++98 -I$(ROOT)/include
LDFLAGS=-L$(ROOT)/core -lcphvb -lrt -shared -Wl,-soname
HEADER=./*.h $(ROOT)/include/*.h
SRC= $(shell find -iname '*.cpp') $(shell find -iname '*.c')
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)
INSTALLDIR?=/opt/cphvb/

all: $(LIB)

%.o: %.c $(HEADER)
		$(CXX) -c $< -o $@ $(LDFLAGS) $(CFLAGS)

%.o: %.cpp $(HEADER)
		$(CXX) -c $< -o $@ $(LDFLAGS) $(CXXFLAGS)

$(LIB): $(OBJ)
		$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS)

clean:
		rm -rf *.o *~ *.so

install: all
		sudo cp $(LIB) $(INSTALLDIR)

