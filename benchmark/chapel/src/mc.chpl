module MonteCarloPi {
    use Random;
    use Regexp;

    config const N = 10000,
                 I = 10,
                 SEED = 589494283;
    config const size = "1024*1024";

    var myRegexp = compile("*");

    writeln(size.split(myRegexp,2));

    proc main() {
        var rs = new RandomStream(SEED, parSafe=false);

        var acc : sync real = 0;
        for i in 1..#I {
            var count = 0;
            for j in 1..#N do {
                if (sqrt(rs.getNext()**2+rs.getNext()**2)<= 1.0) then {
                    count += 1;
                }
            }
            acc += count * 4.0 / N;
        }
        var total_pi = acc / I;
        delete rs;

        writeln(
            "Approximation of pi = ", format("#.######", total_pi), 
            ", N=", N, 
            ", I=", I, 
            "."
        );
    }
}

