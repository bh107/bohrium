% Making sure slicing is working: we expect results to be 99.0000
A = ones(4);
A(1:3, 1:4) = 99;		% fill matrix with 99's except for the bottom row
A(1, 1)
A(4, 1:3) = 99;			% fill bottom row with 99's, except for the last element
A(end, end) = 99		% A(4, 4) = 99 (test end keyword)

B = ones(6);
C = B(1:2, 1:3) * 49.5 + B(1:2, 4:6) * 49.5
C(1:2, 2) = -1;						% update the second column
C(1:2, 1:2:3)						% skip second column

D = B(2:end-1, 2:end-1) + 49 * 2; 	% discard the borders of B
D(1:end, 1:end)						% print D
D(1:2:end, 1:2:end)					% print every other value

B = [1,2,3,4,5];
E = B(1:4) * 25 + [74, 49, 24, -1] 	% one dimension

% Problems:
%C(1:2, 3:-1:1)	negative interval not supported

