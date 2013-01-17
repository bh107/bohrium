#region Copyright
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
#endregion

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL
{
    /// <summary>
    /// A structure for describing view ranges
    /// </summary>
    public struct Range
    {
        /// <summary>
        /// The first index to view in the range
        /// </summary>
        public readonly long First;
        /// <summary>
        /// The last index to view in the range
        /// </summary>
        public readonly long Last;
        /// <summary>
        /// The stride of the elements
        /// </summary>
        public readonly long Stride;
        /// <summary>
        /// A flag that indicates if the struct was initialized
        /// </summary>
        public readonly bool Initialized;
        /// <summary>
        /// A flag that indicates if the struct was initialized with a single index
        /// </summary>
        public readonly bool SingleElement;
        /// <summary>
        /// A flag that indicates that this is a new dimension (of length 1)
        /// </summary>
        public readonly bool NewDimension;

        /// <summary>
        /// Initializes a new instance of the <see cref="Range"/> struct
        /// </summary>
        /// <param name="element">The element to view</param>
        public Range(long element)
        {
            this.First = element;
            this.Last = element;
            this.Stride = 1;
            this.Initialized = true;
            this.SingleElement = true;
            this.NewDimension = false;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Range"/> struct
        /// </summary>
        /// <param name="first">The first index to view</param>
        /// <param name="last">The last index to view</param>
        public Range(long first, long last)
        {
            this.First = first;
            this.Last = last;
            this.Stride = 1;
            this.Initialized = true;
            this.SingleElement = false;
            this.NewDimension = false;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Range"/> struct
        /// </summary>
        /// <param name="first">The first index to view</param>
        /// <param name="last">The last index to view</param>
        /// <param name="stride">The stride in elements of the dimension</param>
        public Range(long first, long last, long stride)
        {
            this.First = first;
            this.Last = last;
            this.Stride = stride;
            this.Initialized = true;
            this.SingleElement = false;
            this.NewDimension = false;
        }

        /// <summary>
        /// Constructs a new range that is optionally a new dimension
        /// </summary>
        /// <param name="newDim">True if this range represents a new dimension, false otherwise</param>
        public Range(bool newDim)
            : this()
        {
            this.Initialized = true;
            this.NewDimension = true;
        }

        /// <summary>
        /// Returns a new instance of the <see cref="Range"/> struct
        /// </summary>
        /// <param name="element">The element to view</param>
        public static Range R(long element) { return new Range(element); }
        /// <summary>
        /// Returns a new instance of the <see cref="Range"/> struct
        /// </summary>
        /// <param name="first">The first index to view</param>
        /// <param name="last">The last index to view</param>
        public static Range R(long first, long last) { return new Range(first, last); }
        /// <summary>
        /// Returns a new instance of the <see cref="Range"/> struct
        /// </summary>
        /// <param name="first">The first index to view</param>
        /// <param name="last">The last index to view</param>
        /// <param name="stride">The stride in elements of the dimension</param>
        public static Range R(long first, long last, long stride) { return new Range(first, last, stride); }

        /// <summary>
        /// Gets a range that represents all elements
        /// </summary>
        public static Range All { get { return new Range(); } }

        /// <summary>
        /// Gets a range that is a new dimension
        /// </summary>
        public static Range NewAxis { get { return new Range(true); } }

        /// <summary>
        /// Gets a range that is a specific element in the dimension
        /// </summary>
        /// <param name="element"></param>
        /// <returns>A slice range</returns>
        public static Range El(long element) { return new Range(element); }

        /// <summary>
        /// Gets a range that is a slice of dimensions
        /// </summary>
        /// <param name="first">The first dimension to include</param>
        /// <param name="last">The last dimension to include</param>
        /// <returns>A slice range</returns>
        public static Range Slice(long first, long last) { return new Range(first, last); }
    }
}
