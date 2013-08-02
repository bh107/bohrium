ROOT=../../..
CXX=g++
#CXX=clang
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -std=c++98 -fPIC -I$(ROOT)/include
LDFLAGS=-L$(ROOT)/core -lbh -lrt -shared -Wl,-soname
HEADER=./*.hpp $(ROOT)/include/*
SRC= $(shell find -iname '*.cpp') $(shell find -iname '*.c')
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)
INSTALLDIR?=/opt/bohrium/

all: $(OBJ)

%.o: %.c $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

%.o: %.cpp $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

install: all
	@-(echo 'Done.')

clean:
	rm -rf *.o *~
