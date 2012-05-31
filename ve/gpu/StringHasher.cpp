/*
 * Copyright 2012 Kenneth Skovhede <kenneth@hexad.dk>
 *
 * This file is part of cphVB <http://code.google.com/p/cphvb/>.
 *
 * cphVB is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cphVB is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with cphVB. If not, see <http://www.gnu.org/licenses/>.
 */
 
 #include <cphvb.h>
 #include <iostream>

size_t string_hasher(std::string str)
{
#if __cplusplus > 199711L
    return std::hash<std::string>(str);
#else
	//fnv_64_str fixed to use fnv-1 initial value
	cphvb_int64 hval = 0xcbf29ce484222325ULL;
    const unsigned char *s = (const unsigned char *)str.c_str();

    while (*s) {

	/* multiply by the 64 bit FNV magic prime mod 2^64 */
	hval += (hval << 1) + (hval << 4) + (hval << 5) +
		(hval << 7) + (hval << 8) + (hval << 40);

	/* xor the bottom with the current octet */
	hval ^= (cphvb_int64)*s++;
    }
	
	return (size_t)hval;
#endif
}