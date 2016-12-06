#!/usr/bin/env python
import sys
import re

def parse(filename):
    lines = {}
    for line in open(filename).readlines():
        m = re.match(".* ([a-z0-9]+_[a-z0-9]+_[a-z0-9]+_[a-z0-9]+), .*", line)
        if m:
            key = m.group(1)
            if not key in lines:
                lines[key] = []
            lines[key].append(line.strip())
    return lines

def main():
    left    = parse(sys.argv[1])
    right   = parse(sys.argv[2])

    both        = sorted(list(set(left.keys()) & set(right.keys())))
    left_only   = sorted(list(set(left.keys()) - set(right.keys())))
    right_only  = sorted(list(set(right.keys()) - set(left.keys())))

    print("Left (shared-kernels)")
    for line in sorted((line for key in both for line in left[key])):
        print line

    print("Right (shared-kernels)")
    for line in sorted((line for key in both for line in right[key])):
        print line

    print("Left (left-only)")
    for line in sorted((line for key in left_only for line in left[key])):
        print line

    print("right (right-only)")
    for line in sorted((line for key in right_only for line in right[key])):
        print line

if __name__ == "__main__":
    main()
