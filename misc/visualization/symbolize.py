#!/usr/bin/env python
#
# This file is part of Bohrium and copyright (c) 2012 the Bohrium team:
# http://cphvb.bitbucket.org
#
# Bohrium is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as 
# published by the Free Software Foundation, either version 3 
# of the License, or (at your option) any later version.
#
# Bohrium is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the 
# GNU Lesser General Public License along with Bohrium. 
#
# If not, see <http://www.gnu.org/licenses/>.
#
import re
import sys

names   = {}
symfst  = 97
symlst  = 122
symcur  = symfst
symrnd  = 0

def lex( line ):

    global names, symfst, symlst, symcur, symrnd
    for m in re.finditer( r'0x[0-9a-f]{6,12}', line ):

        addr = m.group(0)                       # Grab the address from match.
        if addr not in names:                   # Create a symbol for the address.
            if symcur > symlst:
                symcur = symfst
                symrnd += 1
            names[addr] = "%s%d" % (chr(symcur), symrnd) if symrnd > 0 else chr(symcur)
            symcur += 1

        line = line.replace(addr, names[addr])  # Replace address with symbol.
        
    return line

def main():

    for line in sys.stdin:
        print lex(line),

if __name__ == "__main__":
    main()
