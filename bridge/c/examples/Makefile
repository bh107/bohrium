CXX?=gcc
EXEC=helloworld
CFLAGS= -g -O2 -fPIC -I.. -I../../../include/
LDFLAGS= -L.. -lbhc
SRC=helloworld.c
OBJ=$(SRC:.c=.o)

all: $(EXEC)

%.o: %.c ../bh_c.h ../../../include/bh_types.h
	$(CXX) -c $< -o $@ $(LCFLAGS) $(CFLAGS)

$(EXEC): $(OBJ)
	$(CXX) $(OBJ) -o $@ $(LDFLAGS)

clean:
	rm -rf *~ $(EXEC) *.o
