ROOT=../..
include $(ROOT)/platform.mk
LIB=libbh_matmul.$(LIB_SUFFIX)
VISDIR?=/usr/local
CXX?=g++
WARNOPTS=-Wall
LCFLAGS= $(WARNOPTS) $(ARCH_OPTS) -g -O2 -fPIC -I$(ROOT)/include -I$(VISDIR)/include -lGL -lglut -lGLU
LDFLAGS= $(ARCH_OPTS) -L$(ROOT)/core -lbh $(EXTRA_LIBS) -lGL -lglut -lGLU -shared -L$(VISDIR)/lib -lm -Wl,$(LD_LIB_COMMAND)
HEADER=./*.h $(ROOT)/include/*
SRC= $(ALL_CPP_FILES) $(ALL_C_FILES)
OBJTMP=$(SRC:.c=.o)
OBJ=$(OBJTMP:.cpp=.o)
INSTALLDIR?=/opt/bh

all: $(LIB)

%.o: %.c $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)  -I$(ROOT)/include

%.o: %.cpp $(HEADER)
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS) -I$(ROOT)/include

$(LIB): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS),$(LIB) $(CFLAGS) -I$(ROOT)/include

clean:
	rm -rf *.o *~ *.$(LIB_SUFFIX)

install: all
	cp $(LIB) $(INSTALLDIR)
