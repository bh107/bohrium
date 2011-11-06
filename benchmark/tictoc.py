import time

start=0.0

def tic():
    global start
    start = time.time()

def toc():
    global start
    print time.time()-start
    

    
