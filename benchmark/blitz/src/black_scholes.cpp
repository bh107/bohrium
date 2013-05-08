#include <blitz/array.h>
#include <random/uniform.h>
#include <argparse.cpp>

using namespace blitz;
using namespace ranlib;

template <typename T>
Array<T, 1> cnd(Array<T, 1> & x)
{
    int samples = x.numElements();
    Array<T, 1> l(samples), k(samples), w(samples), res(samples);
    Array<bool, 1> mask(samples);
    T a1 = 0.31938153,
      a2 =-0.356563782,
      a3 = 1.781477937,
      a4 =-1.821255978,
      a5 = 1.330274429,
      pp = 2.5066282746310002; // sqrt(2.0*PI)

    l = abs(x);
    k = 1.0 / (1.0 + 0.2316419 * l);
    w = 1.0 - 1.0 / pp * exp(-1.0*l*l/2.0) * \
        (a1*k + \
         a2*(pow(k,(T)2)) + \
         a3*(pow(k,(T)3)) + \
         a4*(pow(k,(T)4)) + \
         a5*(pow(k,(T)5)));

    mask = x < 0.0;
    res = (w * cast<T>(!mask) + (1.0-w)* cast<T>(mask));
    return res;
}

template <typename T>
T* pricing(size_t samples, size_t iterations, char flag, T x, T d_t, T r, T v)
{
    Array<T, 1> d1(samples), d2(samples), res(samples), s(samples);
    T* p    = (T*)malloc(sizeof(T)*samples);    // Intermediate results
    T t     = d_t;                              // Initial delta

    Uniform<T> rand;
    for(size_t i=0; i<samples; i++) {           // Model between 58-62
        s = rand.random() *4.0 -2.0+60.0;
    }

    for(size_t i=0; i<iterations; i++) {
        d1 = (log(s/x) + (r+v*v/2.0)*t) / (v*sqrt(t));
        d2 = d1-v*sqrt(t);
        if (flag == 'c') {
            Array<T, 1> tmp1(samples);
            tmp1 = -r * t;
            res = s * cnd(d1) - x * exp(tmp1) * cnd(d2);
        } else {
            Array<T, 1> tmp1(samples), tmp2(samples), tmp3(samples);
            tmp1 = -r*t;
            tmp2 = -1.0*d2;
            tmp3 = -1.0*d1;

            res = x * exp(tmp1) * cnd(tmp2) - s*cnd(tmp3);
        }

        t += d_t;                               // Increment delta
        p[i] = sum(res) / (T)samples;   // Result from timestep
    }

    return p;
}

int main(int argc, char* argv[])
{
    const char usage[] = "usage: ./black_scholes --size=1000*10 [--verbose]";
    if (2>argc) {
        cout << usage << endl;
        return 1;
    }
    
    arguments_t args;                   // Parse command-line
    if (!parse_args(argc, argv, args)) {
        cout << "Err: Invalid argument(s)." << endl;
        cout << usage << endl;
        return 1;
    }
    if (2 > args.size.size()) {
        cout << "Err: Not enough arguments." << endl;
        cout << usage << endl;
        return 1;
    }
    if (2 < args.size.size()) {
        cout << "Err: Too many arguments." << endl;
        cout << usage << endl;
        return 1;
    }

    double* prices = pricing(           // Do the computations...
        args.size[0], args.size[1],
        'c', 65.0, 1.0 / 365.0,
        0.08, 0.3
    );
                                        // Output timing
    cout << "{elapsed-time: "<< 1000000.0 <<"";          
    if (args.verbose) {                 // and values.
        cout << ", \"output\": [";
        for(size_t i=0; i<args.size[1]; i++) {
            cout << prices[i];
            if (args.size[1]-1!=i) {
                cout << ", ";
            }
        }
        cout << "]" << endl;
    }
    cout << "}" << endl;

    free(prices);                       // Cleanup
    return 0;
}


