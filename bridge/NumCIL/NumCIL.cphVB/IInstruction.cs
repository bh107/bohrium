using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.cphVB
{
    public interface IInstruction
    {
        PInvoke.cphvb_opcode OpCode { get; }
    }
}
