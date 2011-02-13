/*
 * Copyright 2011 Troels Blum <troels@blum.dk>
 *
 * This file is part of cphVB.
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
 * along with cphVB.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "Shape.hpp"
#incluce <cphvb_array.h>

class Shape
{
public:
    Shape(cphvb_int32 ndim, cphvb_index* shape)
    {
        this.ndim = ndim;
        this.shape = shape;
    }

    bool operator==(const Shape &that) const 
    {
        if (this.ndim != that.ndim)
        {
            return false;
        }
        for (int i = 0; i < this.ndim)
        {
            if (this.shape[i] != that.shape[i])
            {
                return false;
            }
        }
        return true;
    }
private:
    cphvb_int32      ndim;
    cphvb_index*     shape;
}
