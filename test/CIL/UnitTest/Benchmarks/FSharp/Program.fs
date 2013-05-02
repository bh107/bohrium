module main

let cmdargs = System.Environment.GetCommandLineArgs()
let mutable bohrium = false
for arg in cmdargs do
   if arg.Equals("--bohrium", System.StringComparison.InvariantCultureIgnoreCase)
   then bohrium <- true         

let mutable version = "CIL"
if bohrium then 
    NumCIL.Bohrium.Utility.Activate()
    version <- "Bohrium"

let timer = new UnitTest.DispTimer("BlackSholes F#, " + version)
let res = BlackScholesSolver.Solve(3200000L, 10)
timer.Dispose()

printfn "result: %f" res