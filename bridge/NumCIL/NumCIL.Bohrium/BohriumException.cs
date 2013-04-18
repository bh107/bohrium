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

namespace NumCIL.Bohrium
{
    /// <summary>
    /// Basic exception class for reporting errors from Bohrium
    /// </summary>
    public class BohriumException : Exception
    {
        /// <summary>
        /// The error code that created the exception
        /// </summary>
        private PInvoke.bh_error m_errorCode;

        /// <summary>
        /// Constructs a new execption
        /// </summary>
        /// <param name="message">The error message to report</param>
        public BohriumException(string message)
            : this(PInvoke.bh_error.BH_ERROR, message)
        {
        }

        /// <summary>
        /// Constructs a new execption
        /// </summary>
        /// <param name="errorcode">The error code to report</param>
        public BohriumException(PInvoke.bh_error errorcode)
            : this(errorcode, string.Format("Bohrium Error: {0}", errorcode))
        { 
        }

        /// <summary>
        /// Constructs a new execption
        /// </summary>
        /// <param name="errorcode">The error code to report</param>
        /// <param name="message">The error message to report</param>
        public BohriumException(PInvoke.bh_error errorcode, string message)
            : base(message)
        {
            m_errorCode = errorcode;
        }

        /// <summary>
        /// Gets the error code the exception was caused by
        /// </summary>
        public PInvoke.bh_error ErrorCode { get { return m_errorCode; } }
    }
}
