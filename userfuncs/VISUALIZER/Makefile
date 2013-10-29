ROOT=../..
LIB=libbh_visualizer.so
VISDIR?=/usr/local
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) -g -O2 -fPIC -I$(ROOT)/include -I$(VISDIR)/include -lGL -lglut -lGLU
LDFLAGS= -L$(ROOT)/core -lbh -lrt -lGL -lglut -lGLU -shared -L$(VISDIR)/lib -lm -Wl,-soname
HEADER=./*.hpp $(ROOT)/include/*
SRC= $(shell find -iname '*.cpp') $(shell find -iname '*.c')
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)
INSTALLDIR?=/opt/bh

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
