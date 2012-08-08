using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

//This namespace contains interfaces for easily recognizing the operators through type matching
namespace NumCIL.Generic.Operators
{
    /// <summary>
    /// This is the addition operator
    /// </summary>
    public interface IAdd { };

    /// <summary>
    /// This is the subtraction operator
    /// </summary>
    public interface ISub { };

    /// <summary>
    /// This is the multiplication operator
    /// </summary>
    public interface IMul { };

    /// <summary>
    /// This is the division operator
    /// </summary>
    public interface IDiv { };

    /// <summary>
    /// This is the modulo operator
    /// </summary>
    public interface IMod { };

    /// <summary>
    /// This is the maxmimum operator
    /// </summary>
    public interface IMax { };

    /// <summary>
    /// This is the minimum operator
    /// </summary>
    public interface IMin { };

    /// <summary>
    /// This is the floor operator
    /// </summary>
    public interface IFloor { };

    /// <summary>
    /// This is the ceiling operator
    /// </summary>
    public interface ICeiling { };

    /// <summary>
    /// This is the rounding operator
    /// </summary>
    public interface IRound { };

    /// <summary>
    /// This is the absolute operator
    /// </summary>
    public interface IAbs { };

    /// <summary>
    /// This is the square root operator
    /// </summary>
    public interface ISqrt { };

    /// <summary>
    /// This is the exponential operator
    /// </summary>
    public interface IExp { };

    /// <summary>
    /// This is the logarithmic operator
    /// </summary>
    public interface ILog { };

    /// <summary>
    /// This is the log10 operator
    /// </summary>
    public interface ILog10 { };

    /// <summary>
    /// This is the power operator
    /// </summary>
    public interface IPow { };

    /// <summary>
    /// This is the cosine operator
    /// </summary>
    public interface ICos { };

    /// <summary>
    /// This is the sine operator
    /// </summary>
    public interface ISin { };

    /// <summary>
    /// This is the tangens operator
    /// </summary>
    public interface ITan { };

    /// <summary>
    /// This is the inverse cosine operator
    /// </summary>
    public interface IAcos { };

    /// <summary>
    /// This is the inverse sine operator
    /// </summary>
    public interface IAsin { };

    /// <summary>
    /// This is the inverse tangens operator
    /// </summary>
    public interface IAtan { };

    /// <summary>
    /// This is the hyperbolic cosine operator
    /// </summary>
    public interface ICosh { };

    /// <summary>
    /// This is the hyperbolic sine operator
    /// </summary>
    public interface ISinh { };

    /// <summary>
    /// This is the hyperbolic tangens operator
    /// </summary>
    public interface ITanh { };

    /// <summary>
    /// This is the and operator
    /// </summary>
    public interface IAnd { };

    /// <summary>
    /// This is the or operator
    /// </summary>
    public interface IOr { };

    /// <summary>
    /// This is the xor operator
    /// </summary>
    public interface IXor { };

    /// <summary>
    /// This is the logical not operator
    /// </summary>
    public interface INot { };

    /// <summary>
    /// This is the bitwise inversion operator
    /// </summary>
    public interface IInvert { };

    /// <summary>
    /// This is the equality operator
    /// </summary>
    public interface IEqual { };

    /// <summary>
    /// This is the greater-than operator
    /// </summary>
    public interface IGreaterThan { };

    /// <summary>
    /// This is the less-than operator
    /// </summary>
    public interface ILessThan { };

    /// <summary>
    /// This is the greater-than-or-equal operator
    /// </summary>
    public interface IGreaterThanOrEqual { };

    /// <summary>
    /// This is the less-than-or-equal operator
    /// </summary>
    public interface ILessThanOrEqual { };

    /// <summary>
    /// This is the inequality operator
    /// </summary>
    public interface INotEqual { };

    /// <summary>
    /// This operation is a type conversion
    /// </summary>
    public interface ITypeConversion { };

    /// <summary>
    /// This operation is a data copy operation
    /// </summary>
    public interface ICopyOperation { };

    /// <summary>
    /// This is the operation that retrives the real component of a complex number
    /// </summary>
    public interface IRealValue { };

    /// <summary>
    /// This is the operation that retrives the imaginary component of a complex number
    /// </summary>
    public interface IImaginaryValue { };
}
