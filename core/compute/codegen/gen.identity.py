#!/usr/bin/env python
import json

all_types = '?bBhHiIlLqQefdgFDGOMm'
all_pairs = [(l,r) for l in all_types for r in all_types]

f_identity = ('CPHVB_IDENTITY', 1, 1, all_pairs)

print json.dumps( f_identity )
