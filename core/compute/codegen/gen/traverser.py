
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
        template_sig = ', '.join(['typename T%d'%op[0] for op in f])
        func_call = []
        for op in f:
            if op[1] == 'a':
                func_call.append('(off%d+d%d)' % (op[0], op[0]))
            else:
                func_call.append('d%d' % op[0])
        t.append({
            'sig': ''.join([op[1] for op in f]),
            'tsig': template_sig,
            'ops': f,
            'func_call': ', '.join(func_call)
        })

    return t

