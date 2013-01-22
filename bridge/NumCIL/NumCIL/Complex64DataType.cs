#region Copyright
/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium:
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

namespace NumCIL.Complex64
{
    /// <summary>
    /// Implementation of a 32+32=64 bit complex number.
    /// Performance is not exceptional in non-Bohrium mode, as anything beyond simple operations are performed with doubles, and truncated back into singles.
    /// The primary use for this class is intended to be for speedups when being used with Bohrium, and GPGPUs in particular.
    /// </summary>
    public struct DataType : IEquatable<DataType>, IFormattable
    {
        /// <summary>
        /// Data storage for the real component
        /// </summary>
        private readonly float m_real;
        /// <summary>
        /// Data storage for the imaginary component
        /// </summary>
        private readonly float m_imag;

        /// <summary>
        /// Gets the real component of the current complex object
        /// </summary>
        public float Real { get { return m_real; } }
        /// <summary>
        /// Gets the imaginary component of the current complex object
        /// </summary>
        public float Imaginary { get { return m_imag; } }

        /// <summary>
        /// Calculates the conjugate value
        /// </summary>
        /// <returns>The conjugate value</returns>
        public DataType Conjugate() { return new DataType(this.Real, -this.Imaginary); }

        /// <summary>
        /// Gets the magnitude value
        /// </summary>
        /// <returns>The magnitude value</returns>
        public float Magnitude { get { return (float)Math.Sqrt((this.Real * this.Real) + (this.Imaginary * this.Imaginary)); } }

        /// <summary>
        /// Gets the phase value
        /// </summary>
        /// <returns>The phase value</returns>
        public float Phase
        {
            get
            {
                if (this.Imaginary == 0 && this.Real < 0)
                    return (float)Math.PI;

                return this.Real >= 0 ? 0.0f : (float)Math.Atan2(this.Imaginary, this.Real);
            }
        }

        /// <summary>
        /// Constructs a new imaginary element
        /// </summary>
        /// <param name="real">The real component</param>
        /// <param name="imag">The imaginary component</param>
        public DataType(float real, float imag)
        {
            m_real = real;
            m_imag = imag;
        }

        #region Mimic complex
        /// <summary>
        /// Returns a new complex instance with a real number equal to zero and an imaginary number equal to one.
        /// </summary>
        public static readonly DataType ImaginaryOne = new DataType(0, 1);
        /// <summary>
        /// Returns a new cComplex instance with a real number equal to one and an imaginary number equal to zero.
        /// </summary>
        public static readonly DataType One = new DataType(1, 0);
        /// <summary>
        /// Returns a new complex instance with a real number equal to zero and an imaginary number equal to zero.
        /// </summary>
        public static readonly DataType Zero = new DataType(0, 0);

        /// <summary>
        /// Returns the additive inverse of a specified complex number.
        /// </summary>
        /// <param name="value">The value to negate</param>
        /// <returns>The result of the complex.Real and complex.Imaginary components of the value parameter multiplied by -1</returns>
        public static DataType operator -(DataType value) { return new DataType(value.Real * -1, value.Imaginary * -1); }
        /// <summary>
        /// Subtracts a complex number from another complex number
        /// </summary>
        /// <param name="left">The value to subtract from (the minuend)</param>
        /// <param name="right">The value to subtract (the subtrahend)</param>
        /// <returns>The result of subtracting right from left</returns>
        public static DataType operator -(DataType left, DataType right) { return new DataType(left.Real - right.Real, left.Imaginary - right.Imaginary); }
        /// <summary>
        /// Returns a value that indicates whether two complex numbers are not equal
        /// </summary>
        /// <param name="left">The first value to compare</param>
        /// <param name="right">The second value to compare</param>
        /// <returns>true if left and right are not equal; otherwise, false</returns>
        public static bool operator !=(DataType left, DataType right) { return !(left == right); }
        /// <summary>
        /// Multiplies two complex numbers
        /// </summary>
        /// <param name="left">The first value to multiply</param>
        /// <param name="right">The second value to multiply</param>
        /// <returns>The product of left and right</returns>
        public static DataType operator *(DataType left, DataType right)
        {
            return new DataType(
                (left.Real * right.Real) - (left.Imaginary * right.Imaginary),
                ((left.Imaginary * right.Real) + (left.Real * right.Imaginary))
            );
        }
        /// <summary>
        /// Divides a complex number by another complex number
        /// </summary>
        /// <param name="left">The value to be divided</param>
        /// <param name="right">The value to divide by</param>
        /// <returns>The result of dividing left by right</returns>
        public static DataType operator /(DataType left, DataType right)
        {
            float d = right.Real * right.Real + right.Imaginary * right.Imaginary;
            return new DataType(
                ((left.Real * right.Real) + (left.Imaginary * right.Imaginary)) / (d),
                ((left.Imaginary * right.Real) - (left.Real * right.Imaginary)) / (d)
            );
        }
        /// <summary>
        /// Adds two complex numbers
        /// </summary>
        /// <param name="left">The first value to add</param>
        /// <param name="right">The second value to add</param>
        /// <returns>The sum of left and right</returns>
        public static DataType operator +(DataType left, DataType right) { return new DataType(left.Real + right.Real, left.Imaginary + right.Imaginary); }
        /// <summary>
        /// Returns a value that indicates whether two complex numbers are equal
        /// </summary>
        /// <param name="left">The first complex number to compare</param>
        /// <param name="right">The second complex number to compare</param>
        /// <returns>true if the left and right parameters have the same value; otherwise, false</returns>
        public static bool operator ==(DataType left, DataType right) { return left.Real == right.Real && left.Imaginary == right.Imaginary; }
        /// <summary>
        /// Defines an explicit conversion of a System.Decimal value to a complex number
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static explicit operator DataType(decimal value) { return new DataType((float)value, 0); }
        /// <summary>
        /// Defines an implicit conversion of an unsigned byte to a complex number
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static implicit operator DataType(byte value) { return new DataType(value, 0); }
        /// <summary>
        /// Defines an implicit conversion of a single-precision floating-point number to a complex number
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static implicit operator DataType(float value) { return new DataType(value, 0); }
        /// <summary>
        /// Defines an implicit conversion of a 32-bit signed integer to a complex number
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static implicit operator DataType(int value) { return new DataType(value, 0); }
        /// <summary>
        /// Defines an implicit conversion of a 64-bit signed integer to a complex number
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static implicit operator DataType(long value) { return new DataType(value, 0); }
        /// <summary>
        /// Defines an implicit conversion of a signed byte to a complex number
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static implicit operator DataType(sbyte value) { return new DataType(value, 0); }
        /// <summary>
        /// Defines an implicit conversion of a 16-bit signed integer to a complex number
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static implicit operator DataType(short value) { return new DataType(value, 0); }
        /// <summary>
        /// Defines an implicit conversion of a 32-bit unsigned integer to a complex number
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static implicit operator DataType(uint value) { return new DataType(value, 0); }
        /// <summary>
        /// Defines an implicit conversion of a 64-bit unsigned integer to a complex number
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static implicit operator DataType(ulong value) { return new DataType(value, 0); }
        /// <summary>
        /// Defines an implicit conversion of a 16-bit unsigned integer to a complex number
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static implicit operator DataType(ushort value) { return new DataType(value, 0); }
        /// <summary>
        /// Defines an implicit conversion of a Complex64 to a Complex128
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static implicit operator System.Numerics.Complex(DataType value) { return new System.Numerics.Complex(value.Real, value.Imaginary); }
        /// <summary>
        /// Defines an explicit conversion of a Complex128 to a complex64
        /// </summary>
        /// <param name="value">The value to convert to a complex number</param>
        /// <returns>A complex number that has a real component equal to value and an imaginary component equal to zero</returns>
        public static explicit operator DataType(System.Numerics.Complex value) { return new DataType((float)value.Real, (float)value.Imaginary); }

        /// <summary>
        /// Gets the absolute value (or magnitude) of a complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The absolute value</returns>
        public static float Abs(DataType value) { return value.Magnitude; }
        /// <summary>
        /// Returns the angle that is the arc cosine of the specified complex number
        /// </summary>
        /// <param name="value">A complex number that represents a cosine</param>
        /// <returns>The angle, measured in radians, which is the arc cosine of value</returns>
        public static DataType Acos(DataType value) { return (DataType)System.Numerics.Complex.Acos(value); }
        /// <summary>
        /// Adds two complex numbers and returns the result
        /// </summary>
        /// <param name="left">The first complex number to add</param>
        /// <param name="right">The second complex number to add</param>
        /// <returns></returns>
        public static DataType Add(DataType left, DataType right) { return left + right; }
        /// <summary>
        /// Returns the angle that is the arc sine of the specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The angle which is the arc sine of value</returns>
        public static DataType Asin(DataType value) { return (DataType)System.Numerics.Complex.Asin(value); }
        /// <summary>
        /// Returns the angle that is the arc tangent of the specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The angle that is the arc tangent of value</returns>
        public static DataType Atan(DataType value) { return (DataType)System.Numerics.Complex.Atan(value); }
        /// <summary>
        /// Computes the conjugate of a complex number and returns the result
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The conjugate of value</returns>
        public static DataType Conjugate(DataType value) { return value.Conjugate(); }
        /// <summary>
        /// Returns the cosine of the specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The cosine of value</returns>
        public static DataType Cos(DataType value) { return (DataType)System.Numerics.Complex.Cos(value); }
        /// <summary>
        /// Returns the hyperbolic cosine of the specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The hyperbolic cosine of value</returns>
        public static DataType Cosh(DataType value) { return (DataType)System.Numerics.Complex.Cosh(value); }
        /// <summary>
        /// Divides one complex number by another and returns the result
        /// </summary>
        /// <param name="dividend">The complex number to be divided</param>
        /// <param name="divisor">The complex number to divide by</param>
        /// <returns>The quotient of the division</returns>
        public static DataType Divide(DataType dividend, DataType divisor) { return dividend / divisor; }
        /// <summary>
        /// Returns a value that indicates whether the current instance and a specified complex number have the same value
        /// </summary>
        /// <param name="value">The complex number to compare</param>
        /// <returns>true if this complex number and value have the same value; otherwise, false</returns>
        public bool Equals(DataType value) { return this == value; }
        /// <summary>
        /// Returns a value that indicates whether the current instance and a specified object have the same value
        /// </summary>
        /// <param name="obj">The object to compare</param>
        /// <returns>true if the obj parameter is a complex object or a type capable of implicit conversion to a complex object, and its value is equal to the current complex object; otherwise, false</returns>
        public override bool Equals(object obj)
        {
            if (obj is DataType)
                return this == ((DataType)obj);
            else
                return false;
        }
        /// <summary>
        /// Returns e raised to the power specified by a complex number
        /// </summary>
        /// <param name="value">A complex number that specifies a power</param>
        /// <returns>The number e raised to the power value</returns>
        public static DataType Exp(DataType value) { return (DataType)System.Numerics.Complex.Exp(value); }
        /// <summary>
        /// Creates a complex number from a point's polar coordinates
        /// </summary>
        /// <param name="magnitude">The magnitude, which is the distance from the origin (the intersection of the x-axis and the y-axis) to the number.</param>
        /// <param name="phase">The phase, which is the angle from the line to the horizontal axis, measured in radians</param>
        /// <returns>A complex number</returns>
        public static DataType FromPolarCoordinates(float magnitude, float phase)
        {
            if (magnitude < 0.0f)
                throw new ArgumentOutOfRangeException("magnitude", "Magnitude must be a positive value");

            return new DataType(magnitude * (float)Math.Cos(phase), magnitude * (float)Math.Sin(phase));
        }
        /// <summary>
        /// Returns the hash code for the current complex object
        /// </summary>
        /// <returns>A 32-bit signed integer hash code</returns>
        public override int GetHashCode() { return this.Real.GetHashCode() ^ (-this.Real.GetHashCode()); }
        /// <summary>
        /// Returns the natural (base e) logarithm of a specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The natural (base e) logarithm of value</returns>
        public static DataType Log(DataType value) { return (DataType)System.Numerics.Complex.Log(value); }
        /// <summary>
        /// Returns the logarithm of a specified complex number in a specified base
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <param name="baseValue">The base of the logarithm</param>
        /// <returns>The logarithm of value in base baseValue</returns>
        public static DataType Log(DataType value, double baseValue) { return (DataType)System.Numerics.Complex.Log(value, baseValue); }
        /// <summary>
        /// Returns the base-10 logarithm of a specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The base-10 logarithm of value</returns>
        public static DataType Log10(DataType value) { return (DataType)System.Numerics.Complex.Log10(value); }
        /// <summary>
        /// Returns the product of two complex numbers
        /// </summary>
        /// <param name="left">The first complex number to multiply</param>
        /// <param name="right">The second complex number to multipl</param>
        /// <returns>The product of the left and right parameters</returns>
        public static DataType Multiply(DataType left, DataType right) { return left * right; }
        /// <summary>
        /// Returns the additive inverse of a specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The result of the complex.Real and complex.Imaginary components of the value parameter multiplied by -1</returns>
        public static DataType Negate(DataType value) { return -value; }
        /// <summary>
        /// Returns a specified complex number raised to a power specified by a complex number
        /// </summary>
        /// <param name="value">A complex number to be raised to a power</param>
        /// <param name="power">A complex number that specifies a power</param>
        /// <returns>The complex number value raised to the power power</returns>
        public static DataType Pow(DataType value, DataType power) { return (DataType)System.Numerics.Complex.Pow(value, power); }
        /// <summary>
        /// Returns a specified complex number raised to a power specified by a single-precision floating-point number
        /// </summary>
        /// <param name="value">A complex number to be raised to a power</param>
        /// <param name="power">A single-precision floating-point number that specifies a power</param>
        /// <returns>The complex number value raised to the power power</returns>
        public static DataType Pow(DataType value, float power) { return (DataType)System.Numerics.Complex.Pow(value, power); }
        /// <summary>
        /// Returns the multiplicative inverse of a complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The reciprocal of value</returns>
        public static DataType Reciprocal(DataType value)
        {
            if (value.Real == 0 && value.Imaginary == 0)
                return Zero;

            return 1 / value;
        }

        /// <summary>
        /// Returns the sine of the specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The sine of value</returns>
        public static DataType Sin(DataType value) { return (DataType)System.Numerics.Complex.Sin(value); }
        /// <summary>
        /// Returns the hyperbolic sine of the specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The hyperbolic sine of value</returns>
        public static DataType Sinh(DataType value) { return (DataType)System.Numerics.Complex.Sinh(value); }
        /// <summary>
        /// Returns the square root of a specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The square root of value</returns>
        public static DataType Sqrt(DataType value) { return (DataType)System.Numerics.Complex.Sqrt(value); }
        /// <summary>
        /// Subtracts one complex number from another and returns the result
        /// </summary>
        /// <param name="left">The value to subtract from (the minuend)</param>
        /// <param name="right">The value to subtract (the subtrahend)</param>
        /// <returns>The result of subtracting right from left</returns>
        public static DataType Subtract(DataType left, DataType right) { return left - right; }
        /// <summary>
        /// Returns the tangent of the specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The tangent of value</returns>
        public static DataType Tan(DataType value) { return (DataType)System.Numerics.Complex.Tan(value); }
        /// <summary>
        /// Returns the hyperbolic tangent of the specified complex number
        /// </summary>
        /// <param name="value">A complex number</param>
        /// <returns>The hyperbolic tangent of value</returns>
        public static DataType Tanh(DataType value) { return (DataType)System.Numerics.Complex.Tanh(value); }
        /// <summary>
        /// Converts the value of the current complex number to its equivalent string representation in Cartesian form
        /// </summary>
        /// <returns>The string representation of the current instance in Cartesian form</returns>
        public override string ToString() { return ((System.Numerics.Complex)this).ToString(); }
        /// <summary>
        /// Converts the value of the current complex number to its equivalent string
        /// </summary>
        /// <param name="provider">An object that supplies culture-specific formatting information</param>
        /// <returns>The string representation of the current instance in Cartesian form, as specified representation in Cartesian form by using the specified culture-specific formatting information by provider</returns>
        public string ToString(IFormatProvider provider) { return ((System.Numerics.Complex)this).ToString(provider); }
        /// <summary>
        /// Converts the value of the current complex number to its equivalent string representation in Cartesian form by using the specified format for its real and imaginary parts
        /// </summary>
        /// <param name="format">A standard or custom numeric format string</param>
        /// <returns>The string representation of the current instance in Cartesian form</returns>
        public string ToString(string format) { return ((System.Numerics.Complex)this).ToString(format); }
        /// <summary>
        /// Converts the value of the current complex number to its equivalent string representation in Cartesian form by using the specified format and culture-specific format information for its real and imaginary parts
        /// </summary>
        /// <param name="format">A standard or custom numeric format string</param>
        /// <param name="provider">An object that supplies culture-specific formatting information</param>
        /// <returns>The string representation of the current instance in Cartesian form, as specified by format and provider</returns>
        public string ToString(string format, IFormatProvider provider) { return ((System.Numerics.Complex)this).ToString(format, provider); }

        #endregion
    }
}
