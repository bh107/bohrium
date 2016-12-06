#include "symbol_table.hpp"
#include "utils.hpp"

using namespace std;

namespace kp{
namespace core{

const char SymbolTable::TAG[] = "SymbolTable";

SymbolTable::SymbolTable(size_t n) : symboltable_()
{
    symboltable_.capacity = n;
    symboltable_.nsymbols = 0;
    symboltable_.table = new kp_operand[symboltable_.capacity];
}

SymbolTable::~SymbolTable(void)
{
    delete[] symboltable_.table;    // De-allocate storage for symbol_table
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
    for(size_t sbl_idx=1; sbl_idx<=symboltable_.nsymbols; ++sbl_idx) {
        ss << prefix << "  [" << sbl_idx << "]{";
        ss << " layout("    << layout_text(symboltable_.table[sbl_idx].layout) << "),";
        ss << " nelem("     << symboltable_.table[sbl_idx].nelem << "),";
        ss << " const_data("<< symboltable_.table[sbl_idx].const_data << "),";
        ss << " etype(" << etype_text(symboltable_.table[sbl_idx].etype) << "),";
        ss << endl << prefix << "  ";
        ss << " ndim("  << symboltable_.table[sbl_idx].ndim << "),";
        ss << " start(" << symboltable_.table[sbl_idx].start << "),";
        ss << " shape(";
        for(int64_t dim_idx=0; dim_idx < symboltable_.table[sbl_idx].ndim; ++dim_idx) {
            ss << symboltable_.table[sbl_idx].shape[dim_idx];
            if (dim_idx != (symboltable_.table[sbl_idx].ndim-1)) {
                ss << prefix << ", ";
            }
        }
        ss << "),";
        ss << " stride(";
        for(int64_t dim_idx=0; dim_idx < symboltable_.table[sbl_idx].ndim; ++dim_idx) {
            ss << symboltable_.table[sbl_idx].stride[dim_idx];
            if (dim_idx != (symboltable_.table[sbl_idx].ndim-1)) {
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

string SymbolTable::text(void)
{
    return text("");
}

void SymbolTable::clear(void)
{
    symboltable_.nsymbols = 0;
}

size_t SymbolTable::capacity(void)
{
    return symboltable_.capacity;
}

size_t SymbolTable::size(void)
{
    return symboltable_.nsymbols;
}

kp_operand& SymbolTable::operator[](size_t operand_idx)
{
    return symboltable_.table[operand_idx];
}

size_t SymbolTable::import(kp_operand& operand)
{
    symboltable_.table[symboltable_.nsymbols++] = operand;
    return symboltable_.nsymbols;
}

size_t SymbolTable::map_operand(bh_instruction& instr, size_t operand_idx)
{
    size_t arg_idx = ++(symboltable_.nsymbols); // Candidate arg_idx if not reused

    if (instr.opcode == BH_RANDOM and operand_idx > 0)
    {
        symboltable_.table[arg_idx].etype        = KP_UINT64;
        if (1 == operand_idx) {
            symboltable_.table[arg_idx].const_data  = &(instr.constant.value.r123.start);
        } else if (2 == operand_idx) {
            symboltable_.table[arg_idx].const_data  = &(instr.constant.value.r123.key);
        } else {
            throw runtime_error("THIS SHOULD NEVER HAPPEN!");
        }
        symboltable_.table[arg_idx].nelem        = 1;
        symboltable_.table[arg_idx].ndim         = 1;
        symboltable_.table[arg_idx].start        = 0;
        symboltable_.table[arg_idx].shape        = instr.operand[operand_idx].shape;
        symboltable_.table[arg_idx].shape[0]     = 1;
        symboltable_.table[arg_idx].stride       = instr.operand[operand_idx].shape;
        symboltable_.table[arg_idx].stride[0]    = 0;
        symboltable_.table[arg_idx].layout       = KP_SCALAR_CONST;
        symboltable_.table[arg_idx].base         = NULL;

    }
    else if (bh_is_constant(&instr.operand[operand_idx])) {  // Constants
        if (BH_R123 != instr.constant.type) {           // Regular constants
            symboltable_.table[arg_idx].const_data   = &(instr.constant.value);
            symboltable_.table[arg_idx].etype        = bhtype_to_etype(instr.constant.type);
        } else {                                        // "Special" for BH_R123
            symboltable_.table[arg_idx].etype        = KP_UINT64;
            if (1 == operand_idx) {
                symboltable_.table[arg_idx].const_data  = &(instr.constant.value.r123.start);
            } else if (2 == operand_idx) {
                symboltable_.table[arg_idx].const_data  = &(instr.constant.value.r123.key);
            } else {
                throw runtime_error("THIS SHOULD NEVER HAPPEN!");
            }
        }
        symboltable_.table[arg_idx].nelem        = 1;
        symboltable_.table[arg_idx].ndim         = 1;
        symboltable_.table[arg_idx].start        = 0;
        symboltable_.table[arg_idx].shape        = instr.operand[operand_idx].shape;
        symboltable_.table[arg_idx].shape[0]     = 1;
        symboltable_.table[arg_idx].stride       = instr.operand[operand_idx].shape;
        symboltable_.table[arg_idx].stride[0]    = 0;
        symboltable_.table[arg_idx].layout       = KP_SCALAR_CONST;
        symboltable_.table[arg_idx].base         = NULL;
    } else {
        symboltable_.table[arg_idx].const_data= NULL;
        symboltable_.table[arg_idx].etype    = bhtype_to_etype(bh_base_array(&instr.operand[operand_idx])->type);
        symboltable_.table[arg_idx].nelem    = bh_base_array(&instr.operand[operand_idx])->nelem;
        symboltable_.table[arg_idx].ndim     = instr.operand[operand_idx].ndim;
        symboltable_.table[arg_idx].start    = instr.operand[operand_idx].start;
        symboltable_.table[arg_idx].shape    = instr.operand[operand_idx].shape;
        symboltable_.table[arg_idx].stride   = instr.operand[operand_idx].stride;

        symboltable_.table[arg_idx].layout   = determine_layout(symboltable_.table[arg_idx]);
        symboltable_.table[arg_idx].base     = (kp_buffer*)instr.operand[operand_idx].base;
    }

    //
    // Reuse kp_operand identifiers: Detect if we have seen it before and reuse the name.
    // This is done by comparing the currently investigated kp_operand (arg_idx)
    // with all other operands in the current scope [1,arg_idx[
    // Do remember that 0 is is not a valid kp_operand and we therefore index from 1.
    // Also we do not want to compare with selv, that is when i == arg_idx.
    for(size_t i=1; i<arg_idx; ++i) {
        if (!equivalent(symboltable_.table[i], symboltable_.table[arg_idx])) {
            continue; // Not equivalent, continue search.
        }
        // Found one! Use it instead of the incremented identifier.
        --symboltable_.nsymbols;
        arg_idx = i;
        break;
    }
    return arg_idx;
}

void SymbolTable::turn_contractable(size_t symbol_idx)
{
    kp_operand & operand = symboltable_.table[symbol_idx];
    if (operand.layout == KP_SCALAR) {
        operand.layout = KP_SCALAR_TEMP;
    } else {
        operand.layout = KP_CONTRACTABLE;
    }

    // If data is already allocated for kp_operand then we do no lower nelem
    // since the nelem is needed by victim-cache to store it... it is important that nelem
    // correctly reflects the amount of elements for which storage is allocated.
    /*
    if (NULL == *kp_operand.data) {
        kp_operand.nelem = 1;
    }
    */
}

kp_operand* SymbolTable::operands(void)
{
    return symboltable_.table;
}


kp_symboltable& SymbolTable::meta(void)
{
    return symboltable_;
}

}}
