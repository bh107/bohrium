/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <vector>

/** zlib wrapper - compress `data`
 *
 * @param data    The data to compress
 * @param nbytes  The number of bytes in data
 * @return        The compressed data
 */
std::vector<unsigned char> zlib_compress(void *data, uint64_t nbytes);

/** zlib wrapper - uncompress `data`
  *
  * @param data         The compressed data
  * @param dest         The destination, which must be large enough for the uncompressed data
  * @param dest_nbytes  The size of the destination buffer
  */
void zlib_uncompress(const std::vector<unsigned char> &data, void *dest, uint64_t dest_nbytes);