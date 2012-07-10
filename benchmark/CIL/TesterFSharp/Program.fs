module main

let cmdargs = System.Environment.GetCommandLineArgs()
let mutable cphvb = false
for arg in cmdargs do
   if arg.Equals("--cphvb", System.StringComparison.InvariantCultureIgnoreCase)
   then cphvb <- true         

let mutable version = "CIL"
if cphvb then 
    NumCIL.cphVB.Utility.Activate()
    version <- "cphVB"

let timer = new UnitTest.DispTimer("BlackSholes F#, " + version)
let res = BlackScholesSolver.Solve(3200000L, 10)
timer.Dispose()

printfn "result: %f" res