var n = 3;
var outerD: domain(2)= {0..n, 0..n};
var innerD: domain(2) ={1..n-1, 1..n-1};

var A, B: [outerD] real;

A[innerD] = B[innerD];
