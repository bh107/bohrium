case ${opcode} + (${op1} << 8) +(${op2} <<12) + (${op3} <<16):
    traverse_${opcount}<${ftypes}, ${fname}_functor<${ftypes}> >( instr );
    break;
