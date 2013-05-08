#ifndef __ARGPARSE
#define __ARGPARSE

#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <getopt.h>

typedef struct arguments {
    std::vector<size_t> size;
    bool verbose;
} arguments_t;

bool parse_args(int argc, char *argv[], arguments_t& args)
{
    int opt = 0;
    int long_index = 0;
    static struct option long_options[] = {
        {"size",    required_argument,  0, 's'},
        {"verbose", optional_argument, 0, 'v'},
        {0,0,0,0}
    };

    args.verbose = false;       // Default: verbose = false
                                
    while ((opt = getopt_long(argc, argv,"sv",
            long_options, &long_index )) != -1) {
        const char *str = optarg;
        switch (opt) {
            case 's':           // Parse --size=N_1*..*N_n
                do {
                    const char *begin = str;
                    while('*' != *str && *str) {
                        str++;
                    }
                    args.size.push_back(atoi(std::string(begin,str).c_str()));
                    
                } while(*str++ != 0);
                break;
            case 'v':           // Parse verbose
                args.verbose = true;
                break;
            default:
                return false;
        }
    }
    return true;
}
#endif

