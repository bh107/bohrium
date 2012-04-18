module BlackScholesSolver

open NumCIL.Float

(* These are two custom NumCIL operators *)
[<Struct>]
type LessThan =
    interface NumCIL.IUnaryOp<float32> with
        member x.Op(a:float32) =
            match a < 0.0f with
            | true -> 
                1.0f
            | false -> 
                0.0f

[<Struct>]
type GreaterThanOrEqual =
    interface NumCIL.IUnaryOp<float32> with
        member x.Op(a:float32) =
            match a >= 0.0f with
            | true -> 
                1.0f
            | false -> 
                0.0f

let CND(X:NdArray) =
    let a1 = 0.31938153f
    let a2 = -0.356563782f
    let a3 = 1.781477937f
    let a4 = -1.821255978f
    let a5 = 1.330274429f

    let L = X.Abs()
    let K = 1.0f / (1.0f + 0.2316419f * L)
    let w = 1.0f - 1.0f / ((float32(sqrt(2.0 * System.Math.PI)))) * (L.Negate() * L / 2.0f).Exp() * (a1 * K + a2 * (K.Pow(2.0f)) + a3 * (K.Pow(3.0f)) + a4 * (K.Pow(4.0f)) + a5 * (K.Pow(5.0f)));
            
    let mask1 = X.Apply<LessThan>()
    let mask2 = X.Apply<GreaterThanOrEqual>()

    w * mask2 + (NdArray(1.0f) - w) * mask1
 

let BlackSholes (callputflag:bool, S:NdArray, X:float32, T:float32, r:float32, v:float32) : NdArray =
    let d1 = ((S / X).Log() + (r + v * v / 2.0f) * T) / (v * (float32(sqrt(T))))
    let d2 = d1 - v * (float32(sqrt(T)))

    let res =
        match callputflag with 
        | true ->
            S * CND(d1) - X * (float32(exp(-r * T))) * CND(d2)
        | false -> 
            X * float32(exp(-r * T)) * CND(d2.Negate()) - S * CND(d1.Negate())

    res

let Solve(size:int64, years:int32) =
    let S = Generate.Random(size)
    let S = S * 4.0f - 2.0f + 60.0f (*Price is 58-62*)

    let X = 65.0f
    let r = 0.08f
    let v = 0.3f

    let day=1.0f/(float32 years)
    let mutable T = day
    let mutable total = 0.0f

    for i in 0..years do
        total <- total + (Add.Reduce(BlackSholes(true, S, X, T, r, v)).Value.[0L] / (float32 size))
        T <- T + day

    total