module Benchmark {
    use Random;

    config const N = 10000,
                 I = 10,
                 SEED = 589494283;

    proc main() {
        var D: domain(1) = 1..N;
        var x,y: [D] real;

        var rs = new RandomStream(SEED, parSafe=false);

        var acc = 0.0;
        for i in 1..I {
            var count = 0;
            for i in 1..N {
                if ((rs.getNext()**2+rs.getNext()**2)<= 1.0) then {
                    count += 1;
                }
            }
            acc += count * 4.0 / N;
            writeln("{Count=", count, ", acc=", acc, "}");
        }
        acc /= I;

        delete rs;
        writeln("Approximation of pi = ", format("#.######", acc));
    }
}

