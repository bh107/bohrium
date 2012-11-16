
def tfunc(ops):
    t = {
        'ops': ops
    }
    return t

def gen( opcodes, ignore ):

    funcs = [
        [(0, 'a'),(1,'a'),(2, 'a')],
        [(0, 'a'),(1,'a'),(2, 'c')],
        [(0, 'a'),(1,'c'),(2, 'a')],
        [(0, 'a'),(1,'a')],
        [(0, 'a'),(1,'c')],
    ]

    t = []
    for f in funcs:
        template_sig = ', '.join(['typename T%d' % op_n for op_n, op_t in f])
        func_call = []
        for op_n, op_t in f:
            if op_t == 'a':
                func_call.append('(off%d+d%d)' % (op_n, op_n))
            else:
                func_call.append('d%d' % op_n)
        t.append({
            'sig': ''.join([op_t for op_n, op_t in f]),
            'tsig': template_sig,
            'ops': f,
            'func_call': ', '.join(func_call)
        })

    return t

