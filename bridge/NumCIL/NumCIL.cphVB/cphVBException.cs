using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.cphVB
{
    class cphVBException : Exception
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
}
