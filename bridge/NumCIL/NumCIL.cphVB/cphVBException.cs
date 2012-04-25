using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.cphVB
{
    public class cphVBException : Exception
    {
        private PInvoke.cphvb_error m_errorCode;
        public cphVBException(string message)
            : this(PInvoke.cphvb_error.CPHVB_ERROR, message)
        {
        }

        public cphVBException(PInvoke.cphvb_error errorcode)
            : this(errorcode, string.Format("CPHVB Error: {0}", errorcode))
        { 
        }

        public cphVBException(PInvoke.cphvb_error errorcode, string message)
            : base(message)
        {
            m_errorCode = errorcode;
        }

        public PInvoke.cphvb_error ErrorCode { get { return m_errorCode; } }
    }

    public class cphVBNotSupportedInstruction : cphVBException
    {
        private long m_instructionNo;
        private PInvoke.cphvb_opcode m_opcode;

        public cphVBNotSupportedInstruction(PInvoke.cphvb_opcode opcode, long instructionNo)
            : base(PInvoke.cphvb_error.CPHVB_PARTIAL_SUCCESS)
        {
            m_opcode = opcode;
            m_instructionNo = instructionNo;
        }

        public long InstructionNo { get { return m_instructionNo; } }
        public PInvoke.cphvb_opcode OpCode { get { return m_opcode; } }
    }
}
