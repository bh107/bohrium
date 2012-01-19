case ${opcode} + (${op1} << 8) +(${op2} <<12):
    traverse_${opcount}<${ftypes}, ${fname}_functor<${ftypes}> >( instr );
    break;
