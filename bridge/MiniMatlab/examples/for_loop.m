% testing for loops: all "ans" should print 1, 50 and 99
l = 1
for i=1:49:99 % with interval
	i
end

l = 2
for i=1:3, % common usage
	i + 48 * (i - 1)
end

l = 3
A = 1; B = 49; C = 99;
for i=A:B:C, % using variables
	i
end

l = 4
for i=-1:-49:-99 % using negative numbers
	i * -1
end

l = 5
for i=1:(24 + 25):(C * 1 + 1 - 1), % using expressions
	i
end
