#include "engine.hpp"
#include "symbol_table.hpp"
#include "dag.hpp"
#include "timevault.hpp"

#include <algorithm>
#include <set>
#include <iomanip>

using namespace std;
using namespace bohrium::core;

namespace bohrium{
namespace engine {
namespace cpu {

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
    const bool dump_rep)
: compiler_cmd(compiler_cmd),
    template_directory(template_directory),
    kernel_directory(kernel_directory),
    object_directory(object_directory),
    vcache_size(vcache_size),
    preload(preload),
    jit_enabled(jit_enabled),
    jit_fusion(jit_fusion),
    jit_dumpsrc(jit_dumpsrc),
    storage(object_directory, kernel_directory),
    specializer(template_directory),
    compiler(compiler_cmd),
    exec_count(0),
    dump_rep(dump_rep)
{
    bh_vcache_init(vcache_size);    // Victim cache
    if (preload) {
        storage.preload();
    }
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
    ss << "}" << endl;
    
    ss << "Attributes {" << endl;
    ss << "  " << specializer.text();    
    ss << "  " << compiler.text();
    ss << "}" << endl;

    return ss.str();    
}

bh_error Engine::sij_mode(SymbolTable& symbol_table, vector<tac_t>& program, Block& block)
{
    bh_error res = BH_SUCCESS;

    tac_t& tac = block.tac(0);
    DEBUG(TAG, tac_text(tac));

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
            // We start by creating a symbol for the block
            block.symbolize();
            //
            // JIT-compile the block if enabled
            if (jit_enabled && \
                (!storage.symbol_ready(block.symbol()))) {   
                                                            // Specialize sourcecode
                string sourcecode = specializer.specialize(symbol_table, block, 0, 0);
                if (jit_dumpsrc==1) {                       // Dump sourcecode to file                
                    core::write_file(
                        storage.src_abspath(block.symbol()),
                        sourcecode.c_str(), 
                        sourcecode.size()
                    );
                }
                TIMER_START
                // Send to compiler
                bool compile_res = compiler.compile(
                    storage.obj_abspath(block.symbol()), 
                    sourcecode.c_str(), 
                    sourcecode.size()
                );
                TIMER_STOP("Compiling (SIJ Kernels)")
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
            TIMER_START
            storage.funcs[block.symbol()](block.operands());
            TIMER_STOP(block.symbol())

            break;
    }
    return BH_SUCCESS;
}

/**
 *  Count temporaries in the 
 *
 */
void count_temp( set<size_t>& disqualified,  set<size_t>& freed,
                 vector<size_t>& reads,  vector<size_t>& writes,
                 set<size_t>& temps) {

    for(set<size_t>::iterator fi=freed.begin(); fi!=freed.end(); ++fi) {
        size_t operand_idx = *fi;
        if ((reads[operand_idx] == 1) && (writes[operand_idx] == 1)) {
            temps.insert(operand_idx);
        }
    }
}

void count_rw(  tac_t& tac, set<size_t>& freed,
                vector<size_t>& reads, vector<size_t>& writes,
                set<size_t>& temps)
{

    switch(tac.op) {    // Do read/write counting ...
        case MAP:
            reads[tac.in1]++;
            writes[tac.out]++;
            break;

        case EXTENSION:
        case ZIP:
            if (tac.in2!=tac.in1) {
                reads[tac.in2]++;
            }
            reads[tac.in1]++;
            writes[tac.out]++;
            break;
        case REDUCE:
        case SCAN:
            reads[tac.in2]++;
            reads[tac.in1]++;
            writes[tac.out]++;
            break;

        case GENERATE:
            switch(tac.oper) {
                case RANDOM:
                case FLOOD:
                    reads[tac.in1]++;
                default:
                    writes[tac.out]++;
            }
            break;

        case NOOP:
        case SYSTEM:    // ... or annotate operands with temp potential.
            if (FREE == tac.oper) {
                freed.insert(tac.out);
            }
            break;
    }
}

bh_error Engine::fuse_mode(SymbolTable& symbol_table, std::vector<tac_t>& program,
                    Dag& graph, size_t subgraph_idx, Block& block)
{
    bh_error res = BH_SUCCESS;

    //
    // Determine temps and fusion_layout
    set<size_t> freed;
    vector<size_t> reads(symbol_table.size()+1);
    fill(reads.begin(), reads.end(), 0);
    vector<size_t> writes(symbol_table.size()+1);
    fill(writes.begin(), writes.end(), 0);
    set<size_t> temps;

    LAYOUT fusion_layout = CONSTANT;
    for(size_t tac_idx=0; tac_idx<block.ntacs(); ++tac_idx) {
        tac_t& tac = block.tac(tac_idx);
        count_rw(tac, freed, reads, writes, temps);

        switch(tac_noperands(tac)) {
            case 3:
                if (symbol_table[tac.in2].layout > fusion_layout) {
                    fusion_layout = symbol_table[tac.in2].layout;
                }
            case 2:
                if (symbol_table[tac.in1].layout > fusion_layout) {
                    fusion_layout = symbol_table[tac.in1].layout;
                }
            case 1:
                if (symbol_table[tac.out].layout > fusion_layout) {
                    fusion_layout = symbol_table[tac.out].layout;
                }
            default:
                break;
        }
    }
    count_temp(symbol_table.disqualified(), freed, reads, writes, temps);

    //
    // Turn temps into scalars
    for(set<size_t>::iterator ti=temps.begin(); ti!=temps.end(); ++ti) {
        symbol_table.turn_scalar(*ti);
    }

    //
    // The operands might have been modified at this point, so we need to create a new symbol.
    if (!block.symbolize()) {
        fprintf(stderr, "Engine::execute(...) == Failed creating symbol.\n");
        return BH_ERROR;
    }

    DEBUG(TAG, "FUSING Subgraph #" << subgraph_idx << " block-symbol=" << block.symbol() << ".");

    //
    // JIT-compile the block if enabled
    //
    if (jit_enabled && \
        (!storage.symbol_ready(block.symbol()))) {   
        // Specialize and dump sourcecode to file
        string sourcecode = specializer.specialize(symbol_table, block, fusion_layout);
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
    if (((graph.omask(subgraph_idx) & (ARRAY_OPS)) >0) && \
        (!storage.symbol_ready(block.symbol())) && \
        (!storage.load(block.symbol()))) {// Need but cannot load

        fprintf(stderr, "Engine::execute(...) == Failed loading object.\n");
        return BH_ERROR;
    }

    //
    // Allocate memory for output
    //
    for(size_t i=0; i<block.ntacs(); ++i) {
        if ((block.tac(i).op & ARRAY_OPS)>0) {

            TIMER_START
            operand_t& operand = symbol_table[block.tac(i).out];
            if ((operand.layout == SCALAR) && \
                (operand.base->data == NULL)) {
                operand.base->nelem = 1;
            }
            bh_error res = bh_vcache_malloc_base(operand.base);
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by bh_vcache_malloc() "
                                "called from bh_ve_cpu_execute()\n");
                return res;
            }
            TIMER_STOP("Allocating memory.")
        }
    }

    //
    // Execute block handling array operations.
    // 
    TIMER_START
    DEBUG(TAG, "Executing...");
    DEBUG(TAG, block.symbol());
    DEBUG(TAG, symbol_table.text("H"));
    storage.funcs[block.symbol()](block.operands());
    TIMER_STOP(block.symbol())

    //
    // De-Allocate operand memory
    for(size_t i=0; i<block.ntacs(); ++i) {
        tac_t& tac = block.tac(i);
        if (tac.oper == FREE) {
            TIMER_START
            res = bh_vcache_free_base(symbol_table[tac.out].base);
            if (BH_SUCCESS != res) {
                fprintf(stderr, "Unhandled error returned by bh_vcache_free(...) "
                                "called from bh_ve_cpu_execute)\n");
                return res;
            }
            TIMER_STOP("Deallocating memory.")
        }
    }

    return BH_SUCCESS;
}

bh_error Engine::execute(bh_instruction* instrs, bh_intp ninstrs)
{
    exec_count++;

    bh_error res = BH_SUCCESS;

    DEBUG(TAG, "0");
    //
    // Instantiate the symbol-table and tac-program
    SymbolTable symbol_table(ninstrs*6+2);              // SymbolTable
    vector<tac_t> program(ninstrs);                     // Program
    DEBUG(TAG, "1");
    // Map instructions to tac and symbol_table
    instrs_to_tacs(instrs, ninstrs, program, symbol_table);
    symbol_table.count_tmp();

    DEBUG(TAG, "2");
    //
    // Construct graph with instructions as nodes.
    Dag graph(symbol_table, program);                   // Graph
    DEBUG(TAG, "3");

    if (dump_rep) {                                     // Dump it to file
        stringstream filename;
        filename << "graph" << exec_count << ".dot";

        std::ofstream fout(filename.str());
        fout << graph.dot() << std::endl;
    }

    DEBUG(TAG, "4(" << graph.subgraphs().size() << ")");
    //
    //  Map subgraphs to blocks one at a time and execute them.
    Block block(symbol_table, program);                 // Block
    for(size_t subgraph_idx=0; subgraph_idx<graph.subgraphs().size(); ++subgraph_idx) {
        Graph& subgraph = *(graph.subgraphs()[subgraph_idx]);

        DEBUG(TAG, "4." << subgraph_idx);
        // FUSE_MODE
        if (jit_fusion && \
            ((graph.omask(subgraph_idx) & (NON_FUSABLE))==0) && \
            ((graph.omask(subgraph_idx) & (ARRAY_OPS)) > 0)) {
            DEBUG(TAG, "4F");
            block.clear();
            block.compose(subgraph);
            fuse_mode(symbol_table, program, graph, subgraph_idx, block);
        } else {
        // SIJ_MODE
            std::pair<vertex_iter, vertex_iter> vip = vertices(subgraph);
            DEBUG(TAG, "4S(" << num_vertices(subgraph) << ")");
            for(vertex_iter vi = vip.first; vi != vip.second; ++vi) {
                // Compose the block
                DEBUG(TAG, "4." << *vi);
                DEBUG(TAG, "4.clear");
                block.clear();
                DEBUG(TAG, "4.compose");
                block.compose(  
                    subgraph.local_to_global(*vi), subgraph.local_to_global(*vi)
                );
                
                DEBUG(TAG, "4.sij");
                // Generate/Load code and execute it
                sij_mode(symbol_table, program, block);
            }
        }
    }
    
    DEBUG(TAG, "5");
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
