#include "symbol_table.hpp"
#include "utils.hpp"

using namespace std;
namespace bohrium{
namespace engine{
namespace cpu{

SymbolTable::SymbolTable(void) : table(NULL), reads(NULL), writes(NULL), reserved(100), nsymbols(0)
{
    init();
}

SymbolTable::SymbolTable(size_t n) : table(NULL), reads(NULL), writes(NULL), reserved(n), nsymbols(0)
{
    init();
}

void SymbolTable::init(void)
{
    table = (operand_t*)malloc(reserved*sizeof(operand_t)); // Storage for symbol_table / operands

    reads   = (size_t*)malloc(reserved*sizeof(size_t));     // Storage for counting reads
    memset(reads, 0, reserved*sizeof(size_t));

    writes  = (size_t*)malloc(reserved*sizeof(size_t));     // Storage for counting writes
    memset(writes, 0, reserved*sizeof(size_t));
}

SymbolTable::~SymbolTable(void)
{
    //
    // De-allocate storage for symbol_table, reads, and writes.
    free(table);
    free(reads);
    free(writes);
    table = NULL;
    reads = NULL;
    writes = NULL;
}

string SymbolTable::text(void)
{
    return text("");
}

string SymbolTable::text_meta(void)
{
    stringstream ss;
    ss << "[";
    ss << "capacity=" << capacity() << ",";
    ss << "size=" << size();
    ss << "]";
    ss << endl;

    return ss.str();
}

string SymbolTable::text(string prefix)
{
    stringstream ss;
    ss << prefix << "symbol_table {" << endl;
    for(size_t sbl_idx=1; sbl_idx<=nsymbols; ++sbl_idx) {
        ss << prefix << "  [" << sbl_idx << "]{";
        ss << " layout("    << utils::layout_text(table[sbl_idx].layout) << "),";
        ss << " nelem("     << table[sbl_idx].nelem << "),";
        ss << " data("      << *(table[sbl_idx].data) << "),";
        ss << " const_data("<< table[sbl_idx].const_data << "),";
        ss << " etype(" << utils::etype_text(table[sbl_idx].etype) << "),";
        ss << endl << prefix << "  ";
        ss << " ndim("  << table[sbl_idx].ndim << "),";
        ss << " start(" << table[sbl_idx].start << "),";        
        ss << " shape(";
        for(int64_t dim_idx=0; dim_idx < table[sbl_idx].ndim; ++dim_idx) {
            ss << table[sbl_idx].shape[dim_idx];
            if (dim_idx != (table[sbl_idx].ndim-1)) {
                ss << prefix << ", ";
            }
        }
        ss << "),";
        ss << " stride(";
        for(int64_t dim_idx=0; dim_idx < table[sbl_idx].ndim; ++dim_idx) {
            ss << table[sbl_idx].stride[dim_idx];
            if (dim_idx != (table[sbl_idx].ndim-1)) {
                ss << prefix << ", ";
            }
        }
        ss << "),";
        ss << " reads(" << reads[sbl_idx] << "),";
        ss << " writes(" << writes[sbl_idx] << ")";
        ss << endl << prefix << "  ";
        ss << "}" << endl;
    }
    ss << prefix << "}" << endl;

    return ss.str();
}

size_t SymbolTable::capacity(void)
{
    return reserved;
}

size_t SymbolTable::size(void)
{
    return nsymbols;
}

operand_t& SymbolTable::operator[](size_t operand_idx)
{
    return table[operand_idx];
}

size_t SymbolTable::map_operand(bh_instruction& instr, size_t operand_idx)
{
    size_t arg_idx = ++(nsymbols);
    if (bh_is_constant(&instr.operand[operand_idx])) {
        table[arg_idx].const_data   = &(instr.constant.value);
        table[arg_idx].data         = &table[arg_idx].const_data;
        table[arg_idx].etype        = utils::bhtype_to_etype(instr.constant.type);
        table[arg_idx].nelem        = 1;
        table[arg_idx].ndim         = 1;
        table[arg_idx].start        = 0;
        table[arg_idx].shape        = instr.operand[operand_idx].shape;
        table[arg_idx].shape[0]     = 1;
        table[arg_idx].stride       = instr.operand[operand_idx].shape;
        table[arg_idx].stride[0]    = 0;
        table[arg_idx].layout       = CONSTANT;
    } else {
        table[arg_idx].const_data= NULL;
        table[arg_idx].data     = &(bh_base_array(&instr.operand[operand_idx])->data);
        table[arg_idx].etype    = utils::bhtype_to_etype(bh_base_array(&instr.operand[operand_idx])->type);
        table[arg_idx].nelem    = bh_base_array(&instr.operand[operand_idx])->nelem;
        table[arg_idx].ndim     = instr.operand[operand_idx].ndim;
        table[arg_idx].start    = instr.operand[operand_idx].start;
        table[arg_idx].shape    = instr.operand[operand_idx].shape;
        table[arg_idx].stride   = instr.operand[operand_idx].stride;

        if (utils::contiguous(table[arg_idx])) {
            table[arg_idx].layout = CONTIGUOUS;
        } else {
            table[arg_idx].layout = STRIDED;
        }
    }

    //
    // Reuse operand identifiers: Detect if we have seen it before and reuse the name.
    // This is done by comparing the currently investigated operand (arg_idx)
    // with all other operands in the current scope [1,arg_idx[
    // Do remember that 0 is is not a valid operand and we therefore index from 1.
    // Also we do not want to compare with selv, that is when i == arg_idx.
    for(size_t i=1; i<arg_idx; ++i) {
        if (!utils::equivalent(table[i], table[arg_idx])) {
            continue; // Not equivalent, continue search.
        }
        // Found one! Use it instead of the incremented identifier.
        --nsymbols;
        arg_idx = i;
        break;
    }
    return arg_idx;
}

void SymbolTable::ref_count(const tac_t& tac)
{
    switch(tac.op) {    // Do read/write counting ...
        case MAP:
            reads[tac.in1]++;
            writes[tac.out]++;
            break;

        case ZIP:
        case EXTENSION:
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

            disqualified.insert(tac.in2);
            disqualified.insert(tac.in1);
            disqualified.insert(tac.out);
            break;

        case GENERATE:
            switch(tac.oper) {
                case RANDOM:
                case FLOOD:
                    reads[tac.in1]++;
                    disqualified.insert(tac.in1);
                default:
                    writes[tac.out]++;
                    disqualified.insert(tac.out);
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

void SymbolTable::turn_scalar(size_t symbol_idx)
{
    operand_t& operand = table[symbol_idx];
    // TODO: Introduce SCALAR as LAYOUT instead of abusing CONSTANT..
    operand.layout = SCALAR;

    //
    // TODO: Hmm in order for this to have effect the victim-cache allocation
    //       needs to be changed... updating the bh_instruction might not be a good idea...
    //       since this only works when the operations are fused... ahh just use malloc_op instead...
    //

    // If data is already allocated for operand then we do no lower nelem
    // since the nelem is needed by victim-cache to store it... it is important that nelem
    // correctly reflects the amount of elements for which storage is allocated.
    if (NULL == *operand.data) {
        operand.nelem = 1;
    }

    //
    // Hmm consider: should be modify the strides?
    //
}

}}}

