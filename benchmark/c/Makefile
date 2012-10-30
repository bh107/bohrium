CC = gcc
CPP = g++
FLAGS = -O2 -lm
BOOST = /opt/local/include/ 

all: jacobi-naive jacobi-simple jacobi-tuned jacobi-vtuned jacobi-boost

bench: all
	./jacobi.sh

clean:
	rm -f jacobi-naive jacobi-simple jacobi-tuned jacobi-vtuned jacobi-boost *~

jacobi-naive: jacobi-naive.c
	$(CC) jacobi-naive.c -o jacobi-naive $(FLAGS)

jacobi-simple: jacobi-simple.c
	$(CC) jacobi-simple.c -o jacobi-simple $(FLAGS)

jacobi-tuned: jacobi-tuned.c
	$(CC) jacobi-tuned.c -o jacobi-tuned $(FLAGS)

jacobi-vtuned: jacobi-vtuned.c
	$(CC) jacobi-vtuned.c -o jacobi-vtuned $(FLAGS)

jacobi-boost: jacobi-boost.cc
	$(CPP) -I$(BOOST) jacobi-boost.cc -o jacobi-boost $(FLAGS)
