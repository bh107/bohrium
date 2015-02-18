module BlackScholesSolver

open NumCIL.Double

let CND(X:NdArray) =
    let a1 = 0.31938153
    let a2 = -0.356563782
    let a3 = 1.781477937
    let a4 = -1.821255978
    let a5 = 1.330274429

    let L = X.Abs()
    let K = 1.0 / (1.0 + 0.2316419 * L)
    let w = 1.0 - 1.0 / ((double(sqrt(2.0 * System.Math.PI)))) * (-L * L / 2.0).Exp() * (a1 * K + a2 * (K.Pow(2.0)) + a3 * (K.Pow(3.0)) + a4 * (K.Pow(4.0)) + a5 * (K.Pow(5.0)));
    
    //F# has a bug/problem/inconsistency, it allows you to define comparison operators
    // but it will not use them, so we have to use this slightly ugly variation: 
    let mask1 = NdArray.op_LessThan(X, 0.0).ToDouble()
    let mask2 = NdArray.op_GreaterThanOrEqual(X, 0.0).ToDouble()

    w * mask2 + (NdArray(1.0) - w) * mask1
 

let BlackSholes (callputflag:bool, S:NdArray, X:double, T:double, r:double, v:double) : NdArray =
    let d1 = ((S / X).Log() + (r + v * v / 2.0) * T) / (v * (double(sqrt(T))))
    let d2 = d1 - v * (double(sqrt(T)))

    let res =
        match callputflag with 
        | true ->
            S * CND(d1) - X * (double(exp(-r * T))) * CND(d2)
        | false -> 
            X * double(exp(-r * T)) * CND(-d2) - S * CND(-d1)

    res

let Solve(size:int64, years:int32) =
    let S = Generate.Random(size)
    let S = S * 4.0 - 2.0 + 60.0 (*Price is 58-62*)

    let X = 65.0
    let r = 0.08
    let v = 0.3

    let day=1.0/(double years)
    let mutable T = day
    let mutable total = 0.0

    for i in 0..years do
        total <- total + (Add.Reduce(BlackSholes(true, S, X, T, r, v)).Value.[0L] / (double size))
        T <- T + day

    total