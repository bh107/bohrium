#include <iomanip>

#include "engine.hpp"
#include "timevault.hpp"
#include "kp_rt.h"
#include "kp_vcache.h"

using namespace std;
using namespace kp::core;

namespace kp{
namespace engine{

const char Engine::TAG[] = "Engine";

Engine::Engine(
    const kp_thread_binding binding,
    const size_t vcache_size,
    const bool preload,
    const bool jit_enabled,
    const bool jit_dumpsrc,
    const bool jit_fusion,
    const bool jit_contraction,
    const bool jit_offload,
    const string compiler_cmd,
    const string compiler_inc,
    const string compiler_lib,
    const string compiler_flg,
    const string compiler_ext,
    const string object_directory,
    const string template_directory,
    const string kernel_directory
    )
:   rt_(NULL),
    preload_(preload),
    jit_enabled_(jit_enabled),
    jit_dumpsrc_(jit_dumpsrc),
    jit_fusion_(jit_fusion),
    jit_contraction_(jit_contraction),
    jit_offload_(jit_offload),
    jit_offload_devid_(jit_offload-1),
    storage_(object_directory, kernel_directory),
    plaid_(template_directory),
    compiler_(compiler_cmd, compiler_inc, compiler_lib, compiler_flg, compiler_ext)
{
    if (preload_) {                 // Object storage
        storage_.preload();
    }

    rt_ = kp_rt_init(vcache_size);      // Initialize CAPE C-runtime
    kp_rt_bind_threads(rt_, binding);   // Bind threads on host PUs

    if (jit_offload_) {             // Add accelerator instance
        Accelerator* accelerator = new Accelerator(jit_offload_devid_);
        if (accelerator->offloadable()) {                           // Verify that it "works"
            accelerators_.push_back(accelerator);
            accelerators_[jit_offload_devid_]->get_max_threads();   // Initialize it
        } else {
            delete accelerator;                                     // Tear it down
            jit_offload_ = false;
            throw runtime_error("Failed initializing accelerator for offload.");
        }
    }

    DEBUG(TAG, text());             // Print the engine configuration
}

Engine::~Engine()
{   
    kp_rt_shutdown(rt_);    // Shut down the CAPE C-runtime

                            // Free accelerator instances
    for(std::vector<Accelerator*>::iterator it=accelerators_.begin();
        it!=accelerators_.end();
        ++it) {
        delete *it;
    }
}

size_t Engine::vcache_size(void)
{
    return kp_rt_vcache_size(rt_);
}

bool Engine::preload(void)
{
    return preload_;
}

bool Engine::jit_enabled(void)
{
    return jit_enabled_;
}

bool Engine::jit_dumpsrc(void)
{
    return jit_dumpsrc_;
}

bool Engine::jit_fusion(void)
{
    return jit_fusion_;
}

bool Engine::jit_contraction(void)
{
    return jit_contraction_;
}

bool Engine::jit_offload(void)
{
    return jit_offload_;
}

int Engine::jit_offload_devid(void)
{
    return jit_offload_devid_;
}

string Engine::text()
{
    stringstream ss;
    ss << "Engine {" << endl;
    ss << "  vcache_size = "        << kp_rt_vcache_size(rt_) << endl;
    ss << "  preload = "            << this->preload_ << endl;    
    ss << "  jit_enabled = "        << this->jit_enabled_ << endl;    
    ss << "  jit_dumpsrc = "        << this->jit_dumpsrc_ << endl;
    ss << "  jit_fusion = "         << this->jit_fusion_ << endl;
    ss << "  jit_contraction = "    << this->jit_contraction_ << endl;
    ss << "  jit_offload = "        << this->jit_offload_ << endl;
    ss << "}" << endl;
    
    ss << storage_.text() << endl;
    ss << compiler_.text() << endl;
    ss << plaid_.text() << endl;

    return ss.str();    
}

bh_error Engine::execute_block(SymbolTable &symbol_table,
                               Program &tac_program,
                               Block &block
)
{
    Accelerator* accelerator = NULL;    // Grab an accelerator instance
    if (jit_offload_) {
        accelerator = accelerators_[0];
    }

    //
    // Buffer Management
    //
    // - allocate output buffer(s) on host
    // - allocate output buffer(s) on accelerator
    // - allocate input buffer(s) on accelerator
    // - push input buffer(s) to accelerator (TODO)
    //
    for(size_t i=0; i<block.ntacs(); ++i) {
        kp_tac & tac = block.tac(i);

        if (!((tac.op & KP_ARRAY_OPS)>0)) {
            continue;
        }
        switch(tac_noperands(tac)) {
            case 3:
                if ((symbol_table[tac.in2].layout & (KP_DYNALLOC_LAYOUT))>0) {
                    if ((accelerator) && (block.iterspace().layout> KP_SCALAR)) {
                        accelerator->alloc(symbol_table[tac.in2]);
                        if (NULL!=symbol_table[tac.in2].base->data) {
                            accelerator->push(symbol_table[tac.in2]);
                        }
                    }
                }
            case 2:
                if ((symbol_table[tac.in1].layout & (KP_DYNALLOC_LAYOUT))>0) {
                    if ((accelerator) && (block.iterspace().layout> KP_SCALAR)) {
                        accelerator->alloc(symbol_table[tac.in1]);
                        if (NULL!=symbol_table[tac.in1].base->data) {
                            accelerator->push(symbol_table[tac.in1]);
                        }
                    }
                }
            case 1:
                if ((symbol_table[tac.out].layout & (KP_DYNALLOC_LAYOUT))>0) {
                    if (!kp_vcache_malloc(symbol_table[tac.out].base)) {
                        fprintf(stderr, "Unhandled error returned by kp_vcache_malloc() "
                                        "called from bh_ve_cpu_execute()\n");
                        return BH_ERROR;
                    }
                    if ((accelerator) && (block.iterspace().layout> KP_SCALAR)) {
                        accelerator->alloc(symbol_table[tac.out]);
                    }
                }
                break;
        }
    }

    //
    // Execute array operations.
    // 
    if (block.narray_tacs() > 0) {
        TIMER_START
        kp_iterspace & iterspace = block.iterspace();   // Grab iteration space
        storage_.funcs[block.symbol()](                 // Execute kernel function
            block.buffers(),
            block.operands(),
            &iterspace,
            jit_offload_devid_
        );
        TIMER_STOP(block.text_compact())
    }

    //
    // Buffer Management
    //
    // - free buffer(s) on accelerator
    // - free buffer(s) on host
    // - pull buffer(s) from accelerator to host
    //
    for(size_t i=0; i<block.ntacs(); ++i) {
        kp_tac & tac = block.tac(i);
        kp_operand & operand = symbol_table[tac.out];

        switch(tac.oper) {  

            case KP_SYNC:               // Pull buffer from accelerator to host
                if (accelerator) {
                    accelerator->pull(operand);
                }
                break;

            case KP_DISCARD:            // Free buffer on accelerator
                if (accelerator) {
                    accelerator->free(operand);
                }
                break;

            case KP_FREE:               // NOTE: Isn't BH_FREE redundant?
                if (accelerator) {      // Free buffer on accelerator
                    accelerator->free(operand);                             // Note: must be done prior to
                }                                                           //       freeing on host.

                if (!kp_vcache_free(operand.base)) {                 // Free buffer on host
                    fprintf(stderr, "Unhandled error returned by kp_vcache_free(...) "
                                    "called from bh_ve_cpu_execute)\n");
                    return BH_ERROR;
                }
                break;

            default:
                break;
        }
    }

    return BH_SUCCESS;
}

bh_error Engine::process_block(SymbolTable &symbol_table,
                               Program &tac_program,
                               Block &block
)
{
    bool consider_jit = jit_enabled_ and (block.narray_tacs() > 0);

    if (!block.symbolize()) {                       // Update block-symbol
        fprintf(stderr, "Engine::process_block(...) == Failed creating symbol.\n");
        return BH_ERROR;
    }

    DEBUG(TAG, "PROCESSING " << block.symbol());

    //
    // JIT-compile: generate source and compile code
    //
    if (consider_jit && \
        (!storage_.symbol_ready(block.symbol()))) {   
        DEBUG(TAG, "JITTING " << block.text());
                                                        // Genereate source
        string sourcecode = codegen::Kernel(plaid_, block).generate_source(jit_offload_);

        bool compile_res;
        if (jit_dumpsrc_==1) {                          // Compile via file
            core::write_file(                           // Dump to file
                storage_.src_abspath(block.symbol()),
                sourcecode.c_str(), 
                sourcecode.size()
            );
            compile_res = compiler_.compile(            // Compile
                storage_.obj_abspath(block.symbol()),
                storage_.src_abspath(block.symbol())
            );
        } else {                                        // Compile via stdin
            compile_res = compiler_.compile(            // Compile
                storage_.obj_abspath(block.symbol()),
                sourcecode.c_str(), 
                sourcecode.size()
            );
        }
        if (!compile_res) {
            fprintf(stderr, "Engine::execute(...) == Compilation failed.\n");

            return BH_ERROR;
        }
        storage_.add_symbol(                            // Inform storage
            block.symbol(),
            storage_.obj_filename(block.symbol())
        );
    }

    //
    // Load the compiled code
    //
    if ((block.narray_tacs() > 0) && \
        (!storage_.symbol_ready(block.symbol())) && \
        (!storage_.load(block.symbol()))) {             // Need but cannot load

        fprintf(stderr, "Engine::execute(...) == Failed loading object.\n");
        return BH_ERROR;
    }

    // Now on with the execution
    return execute_block(symbol_table, tac_program, block);
}

}}

