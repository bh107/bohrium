#include "engine.hpp"
#include "symbol_table.hpp"
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
    const string compiler_cmd,
    const string template_directory,
    const string kernel_directory,
    const string object_directory,
    const size_t vcache_size,
    const bool preload,
    const bool jit_enabled,
    const bool jit_fusion,
    const bool jit_dumpsrc,
    const bool dump_rep,
    const thread_binding binding,
    const size_t mthreads)
: compiler_cmd(compiler_cmd),
    template_directory(template_directory),
    kernel_directory(kernel_directory),
    object_directory(object_directory),
    vcache_size(vcache_size),
    preload(preload),
    jit_enabled(jit_enabled),
    jit_fusion(jit_fusion),
    jit_dumpsrc(jit_dumpsrc),
    dump_rep(dump_rep),
    storage(object_directory, kernel_directory),
    specializer(template_directory),
    compiler(compiler_cmd),
    thread_control(binding, mthreads),
    exec_count(0)
{
    bh_vcache_init(vcache_size);    // Victim cache
    if (preload) {
        storage.preload();
    }
    thread_control.bind_threads();
}

Engine::~Engine()
{   
    if (vcache_size>0) {    // De-allocate the malloc-cache
        bh_vcache_clear();
        bh_vcache_delete();
    }
}

string Engine::text()
{
    stringstream ss;
    ss << "ENVIRONMENT {" << endl;
    ss << "  BH_CORE_VCACHE_SIZE="      << this->vcache_size  << endl;
    ss << "  BH_VE_CPU_PRELOAD="        << this->preload      << endl;    
    ss << "  BH_VE_CPU_JIT_ENABLED="    << this->jit_enabled  << endl;    
    ss << "  BH_VE_CPU_JIT_FUSION="     << this->jit_fusion   << endl;
    ss << "  BH_VE_CPU_JIT_DUMPSRC="    << this->jit_dumpsrc  << endl;
    ss << "  BH_VE_CPU_BIND="           << this->thread_control.get_binding() << endl;
    ss << "  BH_VE_CPU_MTHREADS="       << this->thread_control.get_mthreads() << endl;
    ss << "}" << endl;
    
    ss << "Attributes {" << endl;
    ss << "  " << specializer.text();    
    ss << "  " << compiler.text();
    ss << "}" << endl;

    return ss.str();    
}

bh_error Engine::sij_mode(SymbolTable& symbol_table, vector<tac_t>& program, Block& block)
{
    TIMER_START
    bh_error res = BH_SUCCESS;

    tac_t& tac = block.tac(0);
    switch(tac.op) {
        case NOOP:
            break;

        case SYSTEM:
            switch(tac.oper) {
                case DISCARD:
                case SYNC:
                    break;

                case FREE:
                    res = bh_vcache_free_base(symbol_table[tac.out].base);
                    if (BH_SUCCESS != res) {
                        fprintf(stderr, "Unhandled error returned by bh_vcache_free_base(...) "
                                        "called from Engine::sij_mode)\n");
                        return res;
                    }
                    break;

                default:
                    fprintf(stderr, "Yeah that does not fly...\n");
                    return BH_ERROR;
            }
            break;

        case EXTENSION:
            {
                map<bh_opcode,bh_extmethod_impl>::iterator ext;
                ext = extensions.find(static_cast<bh_instruction*>(tac.ext)->opcode);
                if (ext != extensions.end()) {
                    bh_extmethod_impl extmethod = ext->second;
                    res = extmethod(static_cast<bh_instruction*>(tac.ext), NULL);
                    if (BH_SUCCESS != res) {
                        fprintf(stderr, "Unhandled error returned by extmethod(...) \n");
                        return res;
                    }
                }
            }
            break;

        // Array operations (MAP | ZIP | REDUCE | SCAN)
        case MAP:
        case ZIP:
        case GENERATE:
        case REDUCE:
        case SCAN:

            //
            // We start by creating a symbol for the block and updating the
            // iteration space
            block.symbolize();
            block.update_iterspace();

            //
            // JIT-compile the block if enabled
            if (jit_enabled && \
                (!storage.symbol_ready(block.symbol()))) {   
                                                            // Specialize sourcecode
                string sourcecode = specializer.specialize(symbol_table, block);
                if (jit_dumpsrc==1) {                       // Dump sourcecode to file
                    core::write_file(
                        storage.src_abspath(block.symbol()),
                        sourcecode.c_str(), 
                        sourcecode.size()
                    );
                }
                // Send to compiler
                bool compile_res = compiler.compile(
                    storage.obj_abspath(block.symbol()), 
                    sourcecode.c_str(), 
                    sourcecode.size()
                );
                if (!compile_res) {
                    fprintf(stderr, "Engine::sij_mode(...) == Compilation failed.\n");
                    return BH_ERROR;
                }
                                                            // Inform storage
                storage.add_symbol(block.symbol(), storage.obj_filename(block.symbol()));
            }

            //
            // Load the compiled code
            //
            if ((!storage.symbol_ready(block.symbol())) && \
                (!storage.load(block.symbol()))) {                // Need but cannot load

                fprintf(stderr, "Engine::sij_mode(...) == Failed loading object.\n");
                return BH_ERROR;
            }

            //
            // Allocate memory for operands
            res = bh_vcache_malloc_base(symbol_table[tac.out].base);
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by bh_vcache_malloc_base() "
                                "called from Engine::sij_mode\n");
                return res;
            }

            //
            // Execute block handling array operations.
            // 
            DEBUG(TAG, "EXECUTING " << block.text());
            storage.funcs[block.symbol()](block.operands(), &block.iterspace());

            break;
    }
    TIMER_STOP("S: " + block.symbol())
    return BH_SUCCESS;
}

bh_error Engine::fuse_mode(SymbolTable& symbol_table,
                            std::vector<tac_t>& program,
                            Block& block,
                            bh_ir_kernel& krnl)
{
    bh_error res = BH_SUCCESS;
    TIMER_START

    //
    // Turn temps into scalars
    const std::vector<const bh_base*>& temps = krnl.temp_list();
    for(std::vector<const bh_base*>::const_iterator tmp_it = temps.begin();
        tmp_it != temps.end();
        ++tmp_it) {

        for(size_t operand_idx = 0;
            operand_idx < block.noperands();
            ++operand_idx) {
            if (block.operand(operand_idx).base == *tmp_it) {
                symbol_table.turn_scalar_temp(block.local_to_global(operand_idx));
            }
        }
    }

    //
    // The operands might have been modified at this point, so we need to create a new symbol.
    // and update the iteration-space
    if (!block.symbolize()) {                           // update block-symbol
        fprintf(stderr, "Engine::execute(...) == Failed creating symbol.\n");
        return BH_ERROR;
    }
    block.update_iterspace();                           // update iterspace

    iterspace_t& iterspace = block.iterspace();   // retrieve iterspace

    //
    // JIT-compile the block if enabled
    //
    if (jit_enabled && \
        (!storage.symbol_ready(block.symbol()))) {   
        // Specialize and dump sourcecode to file
        string sourcecode = specializer.specialize(symbol_table, block, iterspace.layout);
        if (jit_dumpsrc==1) {
            core::write_file(
                storage.src_abspath(block.symbol()),
                sourcecode.c_str(), 
                sourcecode.size()
            );
        }
        // Send to compiler
        bool compile_res = compiler.compile(
            storage.obj_abspath(block.symbol()),
            sourcecode.c_str(), 
            sourcecode.size()
        );
        if (!compile_res) {
            fprintf(stderr, "Engine::execute(...) == Compilation failed.\n");

            return BH_ERROR;
        }
        // Inform storage
        storage.add_symbol(block.symbol(), storage.obj_filename(block.symbol()));
    }

    //
    // Load the compiled code
    //
    if ((block.narray_tacs() > 0) && \
        (!storage.symbol_ready(block.symbol())) && \
        (!storage.load(block.symbol()))) {// Need but cannot load

        fprintf(stderr, "Engine::execute(...) == Failed loading object.\n");
        return BH_ERROR;
    }

    //
    // Allocate memory for output operand(s)
    //
    for(size_t i=0; i<block.ntacs(); ++i) {
        tac_t& tac = block.tac(i);
        operand_t& operand = symbol_table[tac.out];

        if (((tac.op & ARRAY_OPS)>0) && (operand.layout!= SCALAR_TEMP)) {
            res = bh_vcache_malloc_base(operand.base);
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                                "called from bh_ve_cpu_execute()\n");
                return res;
            }
        }
    }

    //
    // Execute block handling array operations.
    // 
    DEBUG(TAG, "EXECUTING "<< block.text());
    storage.funcs[block.symbol()](block.operands(), &iterspace);

    //
    // De-Allocate memory for operand(s)
    //
    for(size_t i=0; i<block.ntacs(); ++i) {
        tac_t& tac = block.tac(i);
        operand_t& operand = symbol_table[tac.out];

        if (tac.oper == FREE) {
            res = bh_vcache_free_base(operand.base);
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by bh_vcache_free(...) "
                                "called from bh_ve_cpu_execute)\n");
                return res;
            }
        }
    }
    TIMER_STOP("F: " + block.symbol())
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
    //  Map kernels to blocks one at a time and execute them.
    for(krnl_iter krnl = bhir->kernel_list.begin();
        krnl != bhir->kernel_list.end();
        ++krnl) {

        block.clear();
        block.compose(*krnl);

                
        if (jit_fusion && \
            (block.narray_tacs() > 1)) {                // FUSE_MODE

            DEBUG(TAG, "FUSE START");
            res = fuse_mode(symbol_table, program, block, *krnl);
            if (BH_SUCCESS != res) {
                return res;
            }
            DEBUG(TAG, "FUSE END");
        
        } else {                                        // SIJ_MODE

            DEBUG(TAG, "SIJ START");
            for(std::vector<uint64_t>::iterator idx_it = krnl->instr_indexes.begin();
                idx_it != krnl->instr_indexes.end();
                ++idx_it) {

                // Compose the block
                block.clear();
                block.compose(*idx_it, *idx_it);

                // Generate/Load code and execute it
                res = sij_mode(symbol_table, program, block);
                if (BH_SUCCESS != res) {
                    return res;
                }
            }
            DEBUG(TAG, "SIJ END");
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

    if (extensions.find(opcode) != extensions.end()) {
        fprintf(stderr, "[CPU-VE] Warning, multiple registrations of the same"
               "extension method '%s' (opcode: %d)\n", name, (int)opcode);
    }
    extensions[opcode] = extmethod;

    return BH_SUCCESS;
}

}}}
