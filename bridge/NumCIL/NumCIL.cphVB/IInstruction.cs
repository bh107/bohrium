using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NumCIL.cphVB
{
    /// <summary>
    /// Simple representation of an instruction
    /// </summary>
    public interface IInstruction
    {
        /// <summary>
        /// Gets the opcode this instruction represents
        /// </summary>
        cphvb_opcode OpCode { get; }
    }
}
