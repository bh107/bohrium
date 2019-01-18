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
#include <cstdint>
#include <random>
#include <bhxx/BhArray.hpp>
#include <bhxx/Runtime.hpp>


namespace bhxx {

/** Random class that maintain the state of the random number generation */
class Random {
private:
    uint64_t _seed;
    uint64_t _count = 0;
public:
    /** Create a new random instance
     *
     * @param seed T he seed of the random number generation. If not set, `std::random_device` is used.
     */
    explicit Random(uint64_t seed = std::random_device{}()) : _seed(seed) {}

    /** New 1D random array using the Random123 algorithm <https://www.deshawresearch.com/resources_random123.html>
     *
     * @param size  Size of the new 1D random array
     * @return      The new random array
     */
    BhArray<uint64_t> random123(uint64_t size) {
        BhArray<uint64_t> ret({size});
        Runtime::instance().enqueueRandom(ret, _seed, _count);
        _count += size;
        return ret;
    }

    /** Reset the random instance
     *
     * @param seed  The seed of the random number generation. If not set, `std::random_device` is used.
     */
    void reset(uint64_t seed = std::random_device{}()) {
        _seed = seed;
        _count = 0;
    }

    /** Return random floats in the half-open interval [0.0, 1.0) using Random123
     *
     * @param shape  The shape of the returned array
     * @return       Real array
     */
    template<typename T>
    BhArray<T> randn(Shape shape);
};

/** Exposing the default instance of the random number generation */
extern Random random; // Defined in random.cpp


} // namespace bhxx