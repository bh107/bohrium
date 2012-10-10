from pprint import pprint as pp

def gen( opcodes, ignore ):

    filtered    = [f for f in opcodes if not f['system_opcode'] and f['nop'] > 0 and f['opcode'] not in ignore]
    fname       = [dict(f.items()+{'fname': f['opcode'].lower().replace('cphvb_', '')}.items()) for f in filtered]

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
            if t == "CPHVB_COMPLEX64":
                t = "std::complex<float>"
            elif t == "CPHVB_COMPLEX128":
                t = "std::complex<double>"

            op['ftype'] = t.lower()

            del(op['code'])
            del(op['doc'])
            del(op['system_opcode'])
            del(op['types'])
            if op['nop'] == 3:
                data.append(op)
        
    return data

