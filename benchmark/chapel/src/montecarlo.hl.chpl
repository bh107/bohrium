module MonteCarloPi {
    use Random;
    use Time;
    use Utils;

    config const N, I: int;

    proc main() {
        var D: domain(1) = 1..N;

        var x, y : [D] real;
        var z : [D] bool;

        var t = new Timer();
        t.start();

        var acc = 0.0;
        for i in 1..I {
            fillRandom(x);
            fillRandom(y);
            z = (sqrt(x*x + y*y) <= 1.0);
            acc += (+reduce(z)*4.0)/N;
        }
        acc /= I;
        t.stop();

        writeln("elapsed-timer: ", t.elapsed());
        writeln("Approximation = ", acc);
    }
}

