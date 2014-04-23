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

#include <bh.h>
#include "bh_ve_cpu.h"
#include "engine.hpp"

using namespace std;
const char TAG[] = "Component";

static bh_component myself;

static size_t exec_count = 0;

//
// This is where the actual engine implementation is
static bohrium::engine::cpu::Engine* engine = NULL;

void bh_string_option(char *&option, const char *env_name, const char *conf_name)
{
    option = getenv(env_name);           // For the compiler
    if (NULL==option) {
        option = bh_component_config_lookup(&myself, conf_name);
    }
    char err_msg[100];

    if (!option) {
        sprintf(err_msg, "cpu-ve: String is not set; option (%s).\n", conf_name);
        throw runtime_error(err_msg);
    }
}

void bh_path_option(char *&option, const char *env_name, const char *conf_name)
{
    option = getenv(env_name);           // For the compiler
    if (NULL==option) {
        option = bh_component_config_lookup(&myself, conf_name);
    }
    char err_msg[100];

    if (!option) {
        sprintf(err_msg, "cpu-ve: Path is not set; option (%s).\n", conf_name);
        throw runtime_error(err_msg);
    }
    if (0 != access(option, F_OK)) {
        if (ENOENT == errno) {
            sprintf(err_msg, "cpu-ve: Path does not exist; path (%s).\n", option);
        } else if (ENOTDIR == errno) {
            sprintf(err_msg, "cpu-ve: Path is not a directory; path (%s).\n", option);
        } else {
            sprintf(err_msg, "cpu-ve: Path is broken somehow; path (%s).\n", option);
        }
        throw runtime_error(err_msg);
    }
}

/* Component interface: init (see bh_component.h) */
bh_error bh_ve_cpu_init(const char *name)
{
    DEBUG(TAG,"++ bh_ve_cpu_init(...);");

    bh_intp vcache_size  = 10;  // Default...
    bh_intp jit_enabled  = 1;
    bh_intp jit_preload  = 1;
    bh_intp jit_fusion   = 1;
    bh_intp jit_dumpsrc  = 0;

    char* compiler_cmd;   // cpu Arguments
    char* kernel_path;
    char* object_path;
    char* template_path;

    char *env;
    bh_error err;

    if((err = bh_component_init(&myself, name)) != BH_SUCCESS)
        return err;
    if(myself.nchildren != 0)
    {
        std::cerr << "[CPU-VE] Unexpected number of children, must be 0" << std::endl;
        return BH_ERROR;
    }

    env = getenv("BH_CORE_VCACHE_SIZE");      // Override block_size from environment-variable.
    if (NULL != env) {
        vcache_size = atoi(env);
    }
    if (0 > vcache_size) {                          // Verify it
        fprintf(stderr, "BH_CORE_VCACHE_SIZE (%ld) should be greater than zero!\n", (long int)vcache_size);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_JIT_ENABLED");
    if (NULL != env) {
        jit_enabled = atoi(env);
    }
    if (!((0==jit_enabled) || (1==jit_enabled))) {
        fprintf(stderr, "BH_VE_CPU_JIT_ENABLED (%ld) should 0 or 1.\n", (long int)jit_enabled);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_JIT_PRELOAD");
    if (NULL != env) {
        jit_preload = atoi(env);
    }
    if (!((0==jit_preload) || (1==jit_preload))) {
        fprintf(stderr, "BH_VE_CPU_JIT_PRELOAD (%ld) should 0 or 1.\n", (long int)jit_preload);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_JIT_FUSION");
    if (NULL != env) {
        jit_fusion = atoi(env);
    }
    if (!((0==jit_fusion) || (1==jit_fusion))) {
        fprintf(stderr, "BH_VE_CPU_JIT_FUSION (%ld) should 0 or 1.\n", (long int)jit_fusion);
        return BH_ERROR;
    }

    env = getenv("BH_VE_CPU_JIT_DUMPSRC");
    if (NULL != env) {
        jit_dumpsrc = atoi(env);
    }
    if (!((0==jit_dumpsrc) || (1==jit_dumpsrc))) {
         fprintf(stderr, "BH_VE_CPU_JIT_DUMPSRC (%ld) should 0 or 1.\n", (long int)jit_dumpsrc);
        return BH_ERROR;
    }
        
    // Configuration
    bh_path_option(     kernel_path,    "BH_VE_CPU_KERNEL_PATH",   "kernel_path");
    bh_path_option(     object_path,    "BH_VE_CPU_OBJECT_PATH",   "object_path");
    bh_path_option(     template_path,  "BH_VE_CPU_TEMPLATE_PATH", "template_path");
    bh_string_option(   compiler_cmd,   "BH_VE_CPU_COMPILER",      "compiler_cmd");

    if (!jit_enabled) {
        jit_preload     = 1;
        jit_fusion      = 0;
        jit_dumpsrc     = 0;
    }

	//
    // Make sure that kernel and object path exists
	// TODO: This is anti-portable and should be fixed.
    mkdir(kernel_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    mkdir(object_path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    //
    // VROOM VROOM VROOOOOOMMMM!!! VROOOOM!!
    engine = new bohrium::engine::cpu::Engine(
        string(compiler_cmd),
        string(template_path),
        string(kernel_path),
        string(object_path),
        (size_t)vcache_size,
        (bool)jit_enabled,
        (bool)jit_preload,
        (bool)jit_fusion,
        (bool)jit_dumpsrc
    );

    DEBUG(TAG,"-- bh_ve_cpu_init(...);");
    return BH_SUCCESS;
}

/* Component interface: execute (see bh_component.h) */
bh_error bh_ve_cpu_execute(bh_ir* bhir)
{
    exec_count++;
    return engine->execute(*bhir);
}

/* Component interface: shutdown (see bh_component.h) */
bh_error bh_ve_cpu_shutdown(void)
{
    DEBUG(TAG,"++ bh_ve_cpu_shutdown(void)");
    cout << "Execute got called " << exec_count << " times.";

    bh_component_destroy(&myself);
    
    delete engine;
    engine = NULL;

    DEBUG(TAG,"-- bh_ve_cpu_shutdown(...);");
    return BH_SUCCESS;
}

/* Component interface: extmethod (see bh_component.h) */
bh_error bh_ve_cpu_extmethod(const char *name, bh_opcode opcode)
{
    DEBUG(TAG,"++ bh_ve_cpu_extmethod(...,...)");

    bh_error register_res = engine->register_extension(myself, name, opcode);
    
    return register_res;
}

