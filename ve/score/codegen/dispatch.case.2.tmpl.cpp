case ${opcode} + (${op1} << 4) +(${op2} <<8):
    traverse_${opcount}<${ftypes}, ${fname}_functor<${ftypes}> >( instr );
    break;
