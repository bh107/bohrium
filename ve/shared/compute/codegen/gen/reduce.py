from pprint import pprint as pp

def gen( opcodes, ignore ):

    filtered    = [f for f in opcodes if f['nop'] == 2 and not f['elementwise'] and f['opcode'].endswith("_REDUCE") ]
    fname       = [dict(f.items()+{'fname': f['opcode'][:-len("_REDUCE")].lower().replace('bh_', '')}.items()) for f in filtered]

    data = []
    for f in fname:

        types = list(set([t[1] for t in f['types']]))
        types.sort()

        for t in types:
            
            op = dict(f.items())
            op['op1'] = t        # The type of every operand is the same as the first input-type
            op['op2'] = t
            op['op3'] = t
           
            #Use the C++ complex data type 
            if t == "BH_COMPLEX64":
                t = "std::complex<float>"
            elif t == "BH_COMPLEX128":
                t = "std::complex<double>"

            op['ftype'] = t.lower()

            del(op['code'])
            del(op['doc'])
            del(op['system_opcode'])
            del(op['types'])
            op['opcode'] = op['opcode'][:-len("_REDUCE")]
            data.append(op)
        
    return data

