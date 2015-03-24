#include <sys/time.h>
#include <stdlib.h>
#include <getopt.h>
#include <bp_util.h>

size_t bp_sample_time(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_usec + tv.tv_sec * 1000000;
}

static void parse_size(bp_arguments_type* args, char* arg)
{
    args->nsizes = 0;
    while (1) {
        char *tail;
        int next;

        while ('*'==*arg) {
            arg++;
        }
        if (*arg == 0) {
            break;
        }

        int errno = 0;
        next = strtol (arg, &tail, 0);

        arg = tail;
        args->sizes[args->nsizes] = next;
        args->nsizes++;
    }
}

bp_arguments_type parse_args(int argc, char** argv)
{   
    bp_arguments_type args;

    static int verbose_flag;
    while (1)
    {
        static struct option long_options[] ={
            {"verbose", no_argument,       &verbose_flag, 1},
            {"size",    required_argument, 0, 's'},
            {0, 0, 0, 0}
        };
        int option_index = 0;   // getopt_long stores option index here

        int c = getopt_long(argc, argv, "s:", long_options, &option_index);
        if (c == -1) {
            break;
        }

        switch (c) {
            case 0:
                if (long_options[option_index].flag != 0) {
                    break;
                }
                break;
            case 's':
                parse_size(&args, optarg);
                break;
            case '?':
                break;
            default:
                abort();
        }
    }
    return args;
}

