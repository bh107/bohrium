#include "symbol_table.hpp"
#include "utils.hpp"

using namespace std;
namespace bohrium{
namespace core{

const char SymbolTable::TAG[] = "SymbolTable";

SymbolTable::SymbolTable(size_t n) : table_(NULL), capacity_(n), nsymbols_(0)
{
    table_ = new operand_t[capacity_];
}

SymbolTable::~SymbolTable(void)
{
    //
    // De-allocate storage for symbol_table, reads, and writes.
    delete[] table_;
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
    ss << prefix << "symbol_table_ {" << endl;
    for(size_t sbl_idx=1; sbl_idx<=nsymbols_; ++sbl_idx) {
        ss << prefix << "  [" << sbl_idx << "]{";
        ss << " layout("    << layout_text(table_[sbl_idx].layout) << "),";
        ss << " nelem("     << table_[sbl_idx].nelem << "),";
        ss << " data("      << *(table_[sbl_idx].data) << "),";
        ss << " const_data("<< table_[sbl_idx].const_data << "),";
        ss << " etype(" << etype_text(table_[sbl_idx].etype) << "),";
        ss << endl << prefix << "  ";
        ss << " ndim("  << table_[sbl_idx].ndim << "),";
        ss << " start(" << table_[sbl_idx].start << "),";        
        ss << " shape(";
        for(int64_t dim_idx=0; dim_idx < table_[sbl_idx].ndim; ++dim_idx) {
            ss << table_[sbl_idx].shape[dim_idx];
            if (dim_idx != (table_[sbl_idx].ndim-1)) {
                ss << prefix << ", ";
            }
        }
        ss << "),";
        ss << " stride(";
        for(int64_t dim_idx=0; dim_idx < table_[sbl_idx].ndim; ++dim_idx) {
            ss << table_[sbl_idx].stride[dim_idx];
            if (dim_idx != (table_[sbl_idx].ndim-1)) {
                ss << prefix << ", ";
            }
        }
        ss << "),";
        ss << endl << prefix << "  ";
        ss << "}" << endl;
    }
    ss << prefix << "}" << endl;

    return ss.str();
}

void SymbolTable::clear(void)
{
    nsymbols_ = 0;
}

size_t SymbolTable::capacity(void)
{
    return capacity_;
}

size_t SymbolTable::size(void)
{
    return nsymbols_;
}

operand_t& SymbolTable::operator[](size_t operand_idx)
{
    return table_[operand_idx];
}

size_t SymbolTable::import(operand_t& operand)
{
    table_[nsymbols_++] = operand;
    return nsymbols_;
}

size_t SymbolTable::map_operand(bh_instruction& instr, size_t operand_idx)
{
    size_t arg_idx = ++(nsymbols_);
    if (bh_is_constant(&instr.operand[operand_idx])) {
        table_[arg_idx].const_data   = &(instr.constant.value);
        table_[arg_idx].data         = &table_[arg_idx].const_data;
        table_[arg_idx].etype        = bhtype_to_etype(instr.constant.type);
        table_[arg_idx].nelem        = 1;
        table_[arg_idx].ndim         = 1;
        table_[arg_idx].start        = 0;
        table_[arg_idx].shape        = instr.operand[operand_idx].shape;
        table_[arg_idx].shape[0]     = 1;
        table_[arg_idx].stride       = instr.operand[operand_idx].shape;
        table_[arg_idx].stride[0]    = 0;
        table_[arg_idx].layout       = SCALAR_CONST;
        table_[arg_idx].base         = NULL;
    } else {
        table_[arg_idx].const_data= NULL;
        table_[arg_idx].data     = &(bh_base_array(&instr.operand[operand_idx])->data);
        table_[arg_idx].etype    = bhtype_to_etype(bh_base_array(&instr.operand[operand_idx])->type);
        table_[arg_idx].nelem    = bh_base_array(&instr.operand[operand_idx])->nelem;
        table_[arg_idx].ndim     = instr.operand[operand_idx].ndim;
        table_[arg_idx].start    = instr.operand[operand_idx].start;
        table_[arg_idx].shape    = instr.operand[operand_idx].shape;
        table_[arg_idx].stride   = instr.operand[operand_idx].stride;

        if (contiguous(table_[arg_idx])) {
            table_[arg_idx].layout = CONTIGUOUS;
        } else {
            table_[arg_idx].layout = STRIDED;
        }
        table_[arg_idx].base     = instr.operand[operand_idx].base;
    }

    //
    // Reuse operand identifiers: Detect if we have seen it before and reuse the name.
    // This is done by comparing the currently investigated operand (arg_idx)
    // with all other operands in the current scope [1,arg_idx[
    // Do remember that 0 is is not a valid operand and we therefore index from 1.
    // Also we do not want to compare with selv, that is when i == arg_idx.
    for(size_t i=1; i<arg_idx; ++i) {
        if (!equivalent(table_[i], table_[arg_idx])) {
            continue; // Not equivalent, continue search.
        }
        // Found one! Use it instead of the incremented identifier.
        --nsymbols_;
        arg_idx = i;
        break;
    }
    return arg_idx;
}

void SymbolTable::turn_scalar(size_t symbol_idx)
{
    operand_t& operand = table_[symbol_idx];
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

void SymbolTable::turn_scalar_temp(size_t symbol_idx)
{
    operand_t& operand = table_[symbol_idx];
    operand.layout = SCALAR_TEMP;

    // If data is already allocated for operand then we do no lower nelem
    // since the nelem is needed by victim-cache to store it... it is important that nelem
    // correctly reflects the amount of elements for which storage is allocated.
    if (NULL == *operand.data) {
        operand.nelem = 1;
    }
}

operand_t* SymbolTable::operands(void)
{
    return table_;
}

}}

