using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.cphVB
{
    /// <summary>
    /// Basic exception class for reporting errors from cphVB
    /// </summary>
    public class cphVBException : Exception
    {
        /// <summary>
        /// The error code that created the exception
        /// </summary>
        private PInvoke.cphvb_error m_errorCode;

        /// <summary>
        /// Constructs a new execption
        /// </summary>
        /// <param name="message">The error message to report</param>
        public cphVBException(string message)
            : this(PInvoke.cphvb_error.CPHVB_ERROR, message)
        {
        }

        /// <summary>
        /// Constructs a new execption
        /// </summary>
        /// <param name="errorcode">The error code to report</param>
        public cphVBException(PInvoke.cphvb_error errorcode)
            : this(errorcode, string.Format("CPHVB Error: {0}", errorcode))
        { 
        }

        /// <summary>
        /// Constructs a new execption
        /// </summary>
        /// <param name="errorcode">The error code to report</param>
        /// <param name="message">The error message to report</param>
        public cphVBException(PInvoke.cphvb_error errorcode, string message)
            : base(message)
        {
            m_errorCode = errorcode;
        }

        /// <summary>
        /// Gets the error code the exception was caused by
        /// </summary>
        public PInvoke.cphvb_error ErrorCode { get { return m_errorCode; } }
    }

    /// <summary>
    /// Specialized exception for detecting instructions that are not supported by the VE
    /// </summary>
    public class cphVBNotSupportedInstruction : cphVBException
    {
        /// <summary>
        /// The instruction index
        /// </summary>
        private long m_instructionNo;
        /// <summary>
        /// The opcode that was not supported
        /// </summary>
        private cphvb_opcode m_opcode;

        /// <summary>
        /// Constructs a new exception
        /// </summary>
        /// <param name="opcode">The opcode that was not supported</param>
        /// <param name="instructionNo">The instruction index</param>
        public cphVBNotSupportedInstruction(cphvb_opcode opcode, long instructionNo)
            : base(PInvoke.cphvb_error.CPHVB_PARTIAL_SUCCESS)
        {
            m_opcode = opcode;
            m_instructionNo = instructionNo;
        }

        /// <summary>
        /// Gets the unsupported instruction's index
        /// </summary>
        public long InstructionNo { get { return m_instructionNo; } }
        /// <summary>
        /// Gets the unsupported opcode
        /// </summary>
        public cphvb_opcode OpCode { get { return m_opcode; } }
    }
}
