#define IPOW(r,b,e) {                         \ 
                    r = 1;                    \
                    while (e)                 \
                    {                         \
                        if (e & 1)            \
                            r *= b;           \
                        e >>= 1;              \
                        b *= b;               \
                    }                         \
                    }
