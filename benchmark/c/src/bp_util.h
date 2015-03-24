#ifndef BP_UTIL
typedef struct bp_arguments {
    int nsizes;
    int sizes[16];
    int verbose;
} bp_arguments_type;

bp_arguments_type parse_args(int argc, char** argv);
size_t bp_sample_time(void);
#endif
