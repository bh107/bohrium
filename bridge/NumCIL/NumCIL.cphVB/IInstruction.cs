using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.cphVB
{
    public interface IInstruction
    {
        cphvb_opcode OpCode { get; }
    }
}
