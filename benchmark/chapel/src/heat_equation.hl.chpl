use Time;
config const N, I: int;

var D:          domain(2)    = {0..N,    0..N};
var centerD:    subdomain(D) = {1..N-1, 1..N-1};

var northD:     subdomain(D) = {0..N-2, 1..N-1};
var southD:     subdomain(D) = {2..N,   1..N-1};
var eastD:      subdomain(D) = {1..N-1, 0..N-2};
var westD:      subdomain(D) = {1..N-1, 2..N};

var grid, temp: [D] real;

// Initialize
grid = 0; // No need to do this? Arrays a default-initialized to a certain value?
grid[ ..  , 0..0] = -273.15;
grid[ ..  , N..N] = -273.15;
grid[N..N ,  .. ] = -273.15;
grid[0..0 ,  .. ] =   40;

// Solve
for i in 1..I {
    temp[centerD] = (               grid[northD]  +
                      grid[westD] + grid[centerD] + grid[eastD] +
                                    grid[southD]                ) * 0.2;

    grid[centerD] = temp[centerD];
}
