% Testing matrices: should contain 99.0000 except for R
A = ones(3) * 99
B = zeros(4) + 99
R = rand(5)
C = ones(2, 3) * 99
D = (zeros(4, 2) + 49.5) * 2
E = rand(1, 3) * zeros(1, 3) + 99
R = rand(3) * rand(3) + zeros(3) - ones(3) / rand(3)
F = zeros(2, 6) + 49.5 * -ones(2, 6) * -2
G = ones(1) / 5 + 98.8
