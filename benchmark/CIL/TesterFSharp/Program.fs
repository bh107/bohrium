module main

//NumCIL.cphVB.Utility.Activate()
let timer = new UnitTest.DispTimer("BlackSholes F#")
let res = BlackScholesSolver.Solve(320000L, 36)
timer.Dispose()

printfn "result: %f" res