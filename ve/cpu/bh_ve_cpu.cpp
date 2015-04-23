/*
This file is part of Bohrium and copyright (c) 2012 the Bohrium
team <http://www.bh107.org>.

Bohrium is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as
published by the Free Software Foundation, either version 3
of the License, or (at your option) any later version.

Bohrium is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the
GNU Lesser General Public License along with Bohrium.

If not, see <http://www.gnu.org/licenses/>.
*/
#include <stdexcept>
#include <map>

#include <errno.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/types.h>
#include <sys/stat.h>

#include <boost/algorithm/string/predicate.hpp>

#include <bh.h>
#include "bh_ve_cpu.h"
#include "engine.hpp"
#include "timevault.hpp"

using namespace std;
const char TAG[] = "Component";

static bh_component myself;

//
// This is where the actual engine implementation is
static bohrium::engine::cpu::Engine* engine = NULL;

/**
 *  Grab an option from ENV or config-file and convert it to
 *  and integer within the range [min, max].
 *
 *
 *
 */
bh_error bh_int_option(bh_intp* option, const char* option_name, int min, int max)
{
    char* raw = bh_component_config_lookup(&myself, option_name);
    if (!raw) {
        fprintf(stderr, "parameter(%s) is missing.\n", option_name);
        return BH_ERROR;
    }
    *option = (bh_intp)atoi(raw);
    if ((*option < min) || (*option > max)) {
        fprintf(
            stderr,
            "%s should be within range [%d,%d].\n",
            option_name, min, max
        );
        return BH_ERROR;
    }
    return BH_SUCCESS;
}

bh_error bh_string_option(char*& option, const char* option_name)
{
    option = bh_component_config_lookup(&myself, option_name);
    if (!option) {
        fprintf(stderr, "%s is missing.\n", option_name);
        return BH_ERROR;
    }
    return BH_SUCCESS;
}

bh_error bh_path_option(char*& option, const char* option_name)
{
    option = bh_component_config_lookup(&myself, option_name);

    if (!option) {
        fprintf(stderr, "Path is not set; option (%s).\n", option_name);
        return BH_ERROR;
    }
    if (0 != access(option, F_OK)) {
        if (ENOENT == errno) {
            fprintf(stderr, "Path does not exist; path (%s).\n", option);
        } else if (ENOTDIR == errno) {
            fprintf(stderr, "Path is not a directory; path (%s).\n", option);
        } else {
            fprintf(stderr, "Path is broken somehow; path (%s).\n", option);
        }
        return BH_ERROR;
    }
    return BH_SUCCESS;
}

/* Component interface: init (see bh_component.h) */
bh_error bh_ve_cpu_init(const char *name)
{
    char* env = getenv("BH_FUSE_MODEL");                    // Set the fuse-model
    if (env != NULL) {
        string e(env);
        if (not boost::iequals(e, string("same_shape_generate_1dreduce"))) {
            fprintf(stderr, "[CPU-VE] Warning! unsupported fuse model: '"
                            "', it may not work.\n");
        }
    } else {
        setenv("BH_FUSE_MODEL", "SAME_SHAPE_GENERATE_1DREDUCE", 1);
    }
    if (BH_SUCCESS != bh_component_init(&myself, name)) {   // Initialize engine
        fprintf(stderr, "[CPU-VE] Failed initializing component\n");
        return BH_ERROR;
    }
    if (0 != myself.nchildren) {                            // Check stack
        fprintf(stderr, "[CPU-VE] Unexpected number of children, must be 0\n");
        return BH_ERROR;
    }

    //
    //  Get engine parameters
    //
    bh_intp bind;
    bh_intp thread_limit;
    bh_intp vcache_size;
    bh_intp preload;

    bh_intp jit_level;
    bh_intp jit_dumpsrc;

    char* compiler_cmd = NULL;
    char* compiler_inc = NULL;
    char* compiler_lib = NULL;
    char* compiler_flg = NULL;
    char* compiler_ext = NULL;

    char* object_path = NULL;
    char* template_path = NULL;
    char* kernel_path = NULL;

    if ((BH_SUCCESS!=bh_int_option(&bind, "bind", 0, 2))                        or \
        (BH_SUCCESS!=bh_int_option(&thread_limit, "thread_limit", 0, 2048))     or \
        (BH_SUCCESS!=bh_int_option(&vcache_size, "vcache_size", 0, 100))        or \
        (BH_SUCCESS!=bh_int_option(&preload, "preload", 0, 1))                  or \
        (BH_SUCCESS!=bh_int_option(&jit_level, "jit_level", 0, 3))              or \
        (BH_SUCCESS!=bh_int_option(&jit_dumpsrc, "jit_dumpsrc", 0, 1))          or \
        (BH_SUCCESS!=bh_string_option(compiler_cmd, "compiler_cmd"))            or \
        (BH_SUCCESS!=bh_string_option(compiler_inc, "compiler_inc"))            or \
        (BH_SUCCESS!=bh_string_option(compiler_lib, "compiler_lib"))            or \
        (BH_SUCCESS!=bh_string_option(compiler_flg, "compiler_flg"))            or \
        (BH_SUCCESS!=bh_string_option(compiler_ext, "compiler_ext"))            or \
        (BH_SUCCESS!=bh_path_option(object_path, "object_path"))                or \
        (BH_SUCCESS!=bh_path_option(kernel_path, "kernel_path"))                or \
        (BH_SUCCESS!=bh_path_option(template_path, "template_path"))) {
        return BH_ERROR;
    }

    //
    //  Set JIT-parameters based on JIT-LEVEL
    //
    bh_intp jit_enabled = 0;
    bh_intp jit_fusion = 0;
    bh_intp jit_contraction = 0;
    switch(jit_level) {
        case 0:                     // Disable JIT, rely on preload.
            preload = 1;
            jit_enabled = 0;
            jit_dumpsrc = 0;
            jit_fusion = 0;
            jit_contraction = 0;
            break;

        case 3:                     // SIJ + Fusion + Contraction
            jit_contraction = 1;
        case 2:                     // SIJ + Fusion
            jit_fusion = 1;
        case 1:                     // SIJ
        default:
            jit_enabled = 1;
            break;
    }

	//
    // Make sure that kernel and object path exists
	// TODO: This is anti-portable and should be fixed.
    mkdir(kernel_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(object_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    //
    // VROOM VROOM VROOOOOOMMMM!!! VROOOOM!!
    engine = new bohrium::engine::cpu::Engine(
        (bohrium::engine::cpu::thread_binding)bind,
        (size_t)thread_limit,
        (size_t)vcache_size,
        (bool)preload,
        (bool)jit_enabled,
        (bool)jit_dumpsrc,
        (bool)jit_fusion,
        (bool)jit_contraction,
        string(compiler_cmd),
        string(compiler_inc),
        string(compiler_lib),
        string(compiler_flg),
        string(compiler_ext),
        string(object_path),
        string(template_path),
        string(kernel_path)
    );

    return BH_SUCCESS;
}

/* Component interface: execute (see bh_component.h) */
bh_error bh_ve_cpu_execute(bh_ir* bhir)
{
    bh_error res = engine->execute(bhir);

    return res;
}

/* Component interface: shutdown (see bh_component.h) */
bh_error bh_ve_cpu_shutdown(void)
{
    bh_component_destroy(&myself);

    delete engine;
    engine = NULL;

    return BH_SUCCESS;
}

/* Component interface: extmethod (see bh_component.h) */
bh_error bh_ve_cpu_extmethod(const char *name, bh_opcode opcode)
{
    bh_error register_res = engine->register_extension(myself, name, opcode);

    return register_res;
}

