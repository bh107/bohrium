#include "engine.hpp"
#include "timevault.hpp"

#include <algorithm>
#include <set>
#include <iomanip>

using namespace std;
using namespace bohrium::core;

namespace bohrium{
namespace engine {
namespace cpu {

typedef std::vector<bh_instruction> instr_iter;
typedef std::vector<bh_ir_kernel>::iterator krnl_iter;

const char Engine::TAG[] = "Engine";

Engine::Engine(
    const thread_binding binding,
    const size_t thread_limit,
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
:   vcache_size_(vcache_size),
    preload_(preload),
    jit_enabled_(jit_enabled),
    jit_dumpsrc_(jit_dumpsrc),
    jit_fusion_(jit_fusion),
    jit_contraction_(jit_contraction),
    jit_offload_(jit_offload),
    jit_offload_devid_(jit_offload-1),
    storage_(object_directory, kernel_directory),
    plaid_(template_directory),
    compiler_(compiler_cmd, compiler_inc, compiler_lib, compiler_flg, compiler_ext),
    thread_control_(binding, thread_limit),
    exec_count(0)
{
    bh_vcache_init(vcache_size);    // Victim cache
    if (preload_) {                 // Object storage
        storage_.preload();
    }
    thread_control_.bind_threads(); // Thread control

    if (jit_offload_) {                         // Add accelerator instance, this is just
        accelerators_.push_back(                // a single accelerator for now.
            new Accelerator(jit_offload_devid_) // And defaults to device "0", that is
        );                                      // the first one available.
    }

    DEBUG(TAG, text());             // Print the engine configuration
}

Engine::~Engine()
{   
    if (vcache_size_>0) {   // De-allocate the malloc-cache
        bh_vcache_clear();
        bh_vcache_delete();
    }
                            // Free accelerator instances
    for(std::vector<Accelerator*>::iterator it=accelerators_.begin();
        it!=accelerators_.end();
        ++it) {
        delete *it;
    }
}

string Engine::text()
{
    stringstream ss;
    ss << "Engine {" << endl;
    ss << "  vcache_size = "        << this->vcache_size_ << endl;
    ss << "  preload = "            << this->preload_ << endl;    
    ss << "  jit_enabled = "        << this->jit_enabled_ << endl;    
    ss << "  jit_dumpsrc = "        << this->jit_dumpsrc_ << endl;
    ss << "  jit_fusion = "         << this->jit_fusion_ << endl;
    ss << "  jit_contraction = "    << this->jit_contraction_ << endl;
    ss << "  jit_offload = "        << this->jit_offload_ << endl;
    ss << "}" << endl;
    
    ss << thread_control_.text() << endl;
    ss << storage_.text() << endl;
    ss << compiler_.text() << endl;
    ss << plaid_.text() << endl;

    return ss.str();    
}

bh_error Engine::execute_block(SymbolTable& symbol_table,
                            std::vector<tac_t>& program,
                            Block& block,
                            bh_ir_kernel& krnl
                            )
{
    bh_error res = BH_SUCCESS;

    bool consider_jit = jit_enabled_ and (block.narray_tacs() > 0);

    Accelerator* accelerator = NULL;    // Grab an accelerator instance
    if (jit_offload_) {
        accelerator = accelerators_[0];
    }

    if (!block.symbolize()) {                       // Update block-symbol
        fprintf(stderr, "Engine::execute_block(...) == Failed creating symbol.\n");
        return BH_ERROR;
    }

    DEBUG(TAG, "EXECUTING " << block.symbol());

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

    //
    // Buffer Management
    //
    // - allocate output buffer(s) on host
    // - allocate output buffer(s) on accelerator
    // - allocate input buffer(s) on accelerator
    // - push input buffer(s) to accelerator (TODO)
    //
    for(size_t i=0; i<block.ntacs(); ++i) {
        tac_t& tac = block.tac(i);

        if (!((tac.op & ARRAY_OPS)>0)) {
            continue;
        }
        switch(tac_noperands(tac)) {
            case 3:
                if ((symbol_table[tac.in2].layout & (DYNALLOC_LAYOUT))>0) {
                    if ((accelerator) && (block.iterspace().layout>SCALAR)) {
                        accelerator->alloc(symbol_table[tac.in2]);
                        if (NULL!=symbol_table[tac.in2].base->data) {
                            accelerator->push(symbol_table[tac.in2]);
                        }
                    }
                }
            case 2:
                if ((symbol_table[tac.in1].layout & (DYNALLOC_LAYOUT))>0) {
                    if ((accelerator) && (block.iterspace().layout>SCALAR)) {
                        accelerator->alloc(symbol_table[tac.in1]);
                        if (NULL!=symbol_table[tac.in1].base->data) {
                            accelerator->push(symbol_table[tac.in1]);
                        }
                    }
                }
            case 1:
                if ((symbol_table[tac.out].layout & (DYNALLOC_LAYOUT))>0) {
                    res = bh_vcache_malloc_base(symbol_table[tac.out].base);
                    if (BH_SUCCESS != res) {
                        fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                                        "called from bh_ve_cpu_execute()\n");
                        return res;
                    }
                    if ((accelerator) && (block.iterspace().layout>SCALAR)) {
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
        iterspace_t& iterspace = block.iterspace(); // Grab iteration space
        storage_.funcs[block.symbol()](             // Execute kernel function
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
        tac_t& tac = block.tac(i);
        operand_t& operand = symbol_table[tac.out];

        switch(tac.oper) {  

            case SYNC:              // Pull buffer from accelerator to host
                if (accelerator) {
                    accelerator->pull(operand);
                }
                break;

            case DISCARD:           // Free buffer on accelerator
                if (accelerator) {
                    accelerator->free(operand);
                }
                break;

            case FREE:              // NOTE: Isn't BH_FREE redundant?
                if (accelerator) {   // Free buffer on accelerator
                    accelerator->free(operand);                             // Note: must be done prior to
                }                                                           //       freeing on host.

                res = bh_vcache_free_base(operand.base);    // Free buffer on host
                if (BH_SUCCESS != res) {
                    fprintf(stderr, "Unhandled error returned by bh_vcache_free(...) "
                                    "called from bh_ve_cpu_execute)\n");
                    return res;
                }
                break;

            default:
                break;
        }
    }

    return BH_SUCCESS;
}

bh_error Engine::execute(bh_ir* bhir)
{
    exec_count++;
    DEBUG(TAG, "EXEC #" << exec_count);
    bh_error res = BH_SUCCESS;

    //
    // Instantiate the tac-program and symbol-table
    uint64_t program_size = bhir->instr_list.size();
    vector<tac_t> program(program_size);                // Program
    SymbolTable symbol_table(program_size*6+2);         // SymbolTable
    
    instrs_to_tacs(*bhir, program, symbol_table);       // Map instructions to 
                                                        // tac and symbol_table.

    Block block(symbol_table, program);                 // Construct a block

    //
    //  Map bh_kernels to Blocks one at a time and execute them.
    for(krnl_iter krnl = bhir->kernel_list.begin();
        krnl != bhir->kernel_list.end();
        ++krnl) {

        block.clear();                                  // Reset the block
        block.compose(*krnl, (bool)jit_contraction_);   // Compose it based on kernel
        
        if ((block.omask() & EXTENSION)>0) {            // Extension-Instruction-Execute (EIE)
            tac_t& tac = block.tac(0);
            map<bh_opcode,bh_extmethod_impl>::iterator ext;
            ext = extensions_.find(static_cast<bh_instruction*>(tac.ext)->opcode);
            if (ext != extensions_.end()) {
                bh_extmethod_impl extmethod = ext->second;
                res = extmethod(static_cast<bh_instruction*>(tac.ext), NULL);
                if (BH_SUCCESS != res) {
                    fprintf(stderr, "Unhandled error returned by extmethod(...) \n");
                    return res;
                }
            }
        } else if ((jit_fusion_) || 
                   (block.narray_tacs() == 0)) {        // Multi-Instruction-Execute (MIE)
            DEBUG(TAG, "Multi-Instruction-Execute BEGIN");
            res = execute_block(symbol_table, program, block, *krnl);
            if (BH_SUCCESS != res) {
                return res;
            }
            DEBUG(TAG, "Muilti-Instruction-Execute END");
        } else {                                        // Single-Instruction-Execute (SIE)
            DEBUG(TAG, "Single-Instruction-Execute BEGIN");
            for(std::vector<uint64_t>::iterator idx_it = krnl->instr_indexes.begin();
                idx_it != krnl->instr_indexes.end();
                ++idx_it) {

                block.clear();                          // Reset the block
                block.compose(*krnl, (size_t)*idx_it);  // Compose based on a single instruction

                res = execute_block(symbol_table, program, block, *krnl);
                if (BH_SUCCESS != res) {
                    return res;
                }
            }
            DEBUG(TAG, "Single-Instruction-Execute END");
        }
    }
    return res;
}

bh_error Engine::register_extension(bh_component& instance, const char* name, bh_opcode opcode)
{
    bh_extmethod_impl extmethod;
    bh_error err = bh_component_extmethod(&instance, name, &extmethod);
    if (err != BH_SUCCESS) {
        return err;
    }

    if (extensions_.find(opcode) != extensions_.end()) {
        fprintf(stderr, "[CPU-VE] Warning, multiple registrations of the same"
               "extension method '%s' (opcode: %d)\n", name, (int)opcode);
    }
    extensions_[opcode] = extmethod;

    return BH_SUCCESS;
}

}}}
