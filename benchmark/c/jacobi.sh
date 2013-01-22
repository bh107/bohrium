#! /bin/tcsh
echo "--- C NAIVE---"
foreach i (1 2 3 4 5)
    ./jacobi-naive 7000 4 | grep time | awk '{print $6}'
end
echo "--- C SIMPLE---"
foreach i (1 2 3 4 5)
    ./jacobi-simple 7000 4 | grep time | awk '{print $6}'
end
echo "--- C TUNED---"
foreach i (1 2 3 4 5)
    ./jacobi-tuned 7000 4 | grep time | awk '{print $6}'
end
echo "--- C VERY TUNED---"
echo "Jacobi"
foreach i (1 2 3 4 5)
    ./jacobi-vtuned 7000 4 | grep time | awk '{print $6}'
end
echo "--- C BOOST---"
echo "Jacobi"
foreach i (1 2 3 4 5)
    ./jacobi-boost 7000 4 | grep time | awk '{print $6}'
end
echo "--- Python/NumPy---"
foreach i (1 2 3 4 5)
    python jacobi.py 7000 4 False
end
echo "--- Python/NumPy/Bohrium---"
foreach i (1 2 3 4 5)
    python jacobi.py 7000 4 True
end

