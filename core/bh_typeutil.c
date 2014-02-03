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
#include <bh.h>

/**
 *  Deduct a numerical value representing the types of the given instruction.
 *
 *  @param instr The instruction for which to deduct a signature.
 *  @return The deducted signature.
 */
int bh_typesig(bh_instruction *instr)
{
    int typesig;
    const int nops = bh_operands(instr->opcode);
    switch(nops) {
        case 3:
            typesig = instr->operand[0].base->type+1;

            if (bh_is_constant(&instr->operand[1])) {
                typesig += ((1+instr->constant.type) << 4) \
                          +((1+instr->operand[2].base->type) << 8);

            } else if (bh_is_constant(&instr->operand[2])) {
                typesig += ((1+instr->operand[1].base->type) << 4) \
                          +((1+instr->constant.type) << 8);

            } else {
                typesig += ((1+instr->operand[1].base->type) << 4) \
                          +((1+instr->operand[2].base->type) << 8);
            }
            break;
        case 2:
            typesig = instr->operand[0].base->type+1;

            if (bh_is_constant(&instr->operand[1])) {
                typesig += ((1+instr->constant.type) << 4);
            } else {
                typesig += ((1+instr->operand[1].base->type) << 4);
            }
            break;
        case 1:
            typesig = (1+instr->operand[0].base->type);
            break;
        case 0:
        default:
            typesig = 0;
            break;
    }

    return typesig;
}

/**
 *  Determine whether the given typesig, in the coding produced by bh_typesig, is valid.
 *
 *  @param instr The instruction for which to deduct a signature.
 *  @return The deducted signature.
 */
bool bh_typesig_check(int typesig)
{
    switch(typesig) {
        case 273: return true; // zzz: BH_BOOL + (BH_BOOL << 4) + (BH_BOOL << 8)
        case 3549: return true; // CCC: BH_COMPLEX128 + (BH_COMPLEX128 << 4) + (BH_COMPLEX128 << 8)
        case 3276: return true; // ccc: BH_COMPLEX64 + (BH_COMPLEX64 << 4) + (BH_COMPLEX64 << 8)
        case 2730: return true; // fff: BH_FLOAT32 + (BH_FLOAT32 << 4) + (BH_FLOAT32 << 8)
        case 3003: return true; // ddd: BH_FLOAT64 + (BH_FLOAT64 << 4) + (BH_FLOAT64 << 8)
        case 819: return true; // sss: BH_INT16 + (BH_INT16 << 4) + (BH_INT16 << 8)
        case 1092: return true; // iii: BH_INT32 + (BH_INT32 << 4) + (BH_INT32 << 8)
        case 1365: return true; // lll: BH_INT64 + (BH_INT64 << 4) + (BH_INT64 << 8)
        case 546: return true; // bbb: BH_INT8 + (BH_INT8 << 4) + (BH_INT8 << 8)
        case 1911: return true; // SSS: BH_UINT16 + (BH_UINT16 << 4) + (BH_UINT16 << 8)
        case 2184: return true; // III: BH_UINT32 + (BH_UINT32 << 4) + (BH_UINT32 << 8)
        case 2457: return true; // LLL: BH_UINT64 + (BH_UINT64 << 4) + (BH_UINT64 << 8)
        case 1638: return true; // BBB: BH_UINT8 + (BH_UINT8 << 4) + (BH_UINT8 << 8)
        case 2721: return true; // zff: BH_BOOL + (BH_FLOAT32 << 4) + (BH_FLOAT32 << 8)
        case 2993: return true; // zdd: BH_BOOL + (BH_FLOAT64 << 4) + (BH_FLOAT64 << 8)
        case 817: return true; // zss: BH_BOOL + (BH_INT16 << 4) + (BH_INT16 << 8)
        case 1089: return true; // zii: BH_BOOL + (BH_INT32 << 4) + (BH_INT32 << 8)
        case 1361: return true; // zll: BH_BOOL + (BH_INT64 << 4) + (BH_INT64 << 8)
        case 545: return true; // zbb: BH_BOOL + (BH_INT8 << 4) + (BH_INT8 << 8)
        case 1905: return true; // zSS: BH_BOOL + (BH_UINT16 << 4) + (BH_UINT16 << 8)
        case 2177: return true; // zII: BH_BOOL + (BH_UINT32 << 4) + (BH_UINT32 << 8)
        case 2449: return true; // zLL: BH_BOOL + (BH_UINT64 << 4) + (BH_UINT64 << 8)
        case 1633: return true; // zBB: BH_BOOL + (BH_UINT8 << 4) + (BH_UINT8 << 8)
        case 3537: return true; // zCC: BH_BOOL + (BH_COMPLEX128 << 4) + (BH_COMPLEX128 << 8)
        case 3265: return true; // zcc: BH_BOOL + (BH_COMPLEX64 << 4) + (BH_COMPLEX64 << 8)
        case 1297: return true; // zzl: BH_BOOL + (BH_BOOL << 4) + (BH_INT64 << 8)
        case 1501: return true; // CCl: BH_COMPLEX128 + (BH_COMPLEX128 << 4) + (BH_INT64 << 8)
        case 1484: return true; // ccl: BH_COMPLEX64 + (BH_COMPLEX64 << 4) + (BH_INT64 << 8)
        case 1450: return true; // ffl: BH_FLOAT32 + (BH_FLOAT32 << 4) + (BH_INT64 << 8)
        case 1467: return true; // ddl: BH_FLOAT64 + (BH_FLOAT64 << 4) + (BH_INT64 << 8)
        case 1331: return true; // ssl: BH_INT16 + (BH_INT16 << 4) + (BH_INT64 << 8)
        case 1348: return true; // iil: BH_INT32 + (BH_INT32 << 4) + (BH_INT64 << 8)
        case 1314: return true; // bbl: BH_INT8 + (BH_INT8 << 4) + (BH_INT64 << 8)
        case 1399: return true; // SSl: BH_UINT16 + (BH_UINT16 << 4) + (BH_INT64 << 8)
        case 1416: return true; // IIl: BH_UINT32 + (BH_UINT32 << 4) + (BH_INT64 << 8)
        case 1433: return true; // LLl: BH_UINT64 + (BH_UINT64 << 4) + (BH_INT64 << 8)
        case 1382: return true; // BBl: BH_UINT8 + (BH_UINT8 << 4) + (BH_INT64 << 8)
        case 17: return true; // zz: BH_BOOL + (BH_BOOL << 4)
        case 170: return true; // ff: BH_FLOAT32 + (BH_FLOAT32 << 4)
        case 187: return true; // dd: BH_FLOAT64 + (BH_FLOAT64 << 4)
        case 51: return true; // ss: BH_INT16 + (BH_INT16 << 4)
        case 68: return true; // ii: BH_INT32 + (BH_INT32 << 4)
        case 85: return true; // ll: BH_INT64 + (BH_INT64 << 4)
        case 34: return true; // bb: BH_INT8 + (BH_INT8 << 4)
        case 119: return true; // SS: BH_UINT16 + (BH_UINT16 << 4)
        case 136: return true; // II: BH_UINT32 + (BH_UINT32 << 4)
        case 153: return true; // LL: BH_UINT64 + (BH_UINT64 << 4)
        case 102: return true; // BB: BH_UINT8 + (BH_UINT8 << 4)
        case 221: return true; // CC: BH_COMPLEX128 + (BH_COMPLEX128 << 4)
        case 204: return true; // cc: BH_COMPLEX64 + (BH_COMPLEX64 << 4)
        case 161: return true; // zf: BH_BOOL + (BH_FLOAT32 << 4)
        case 177: return true; // zd: BH_BOOL + (BH_FLOAT64 << 4)
        case 49: return true; // zs: BH_BOOL + (BH_INT16 << 4)
        case 65: return true; // zi: BH_BOOL + (BH_INT32 << 4)
        case 81: return true; // zl: BH_BOOL + (BH_INT64 << 4)
        case 33: return true; // zb: BH_BOOL + (BH_INT8 << 4)
        case 113: return true; // zS: BH_BOOL + (BH_UINT16 << 4)
        case 129: return true; // zI: BH_BOOL + (BH_UINT32 << 4)
        case 145: return true; // zL: BH_BOOL + (BH_UINT64 << 4)
        case 97: return true; // zB: BH_BOOL + (BH_UINT8 << 4)
        case 29: return true; // Cz: BH_COMPLEX128 + (BH_BOOL << 4)
        case 205: return true; // Cc: BH_COMPLEX128 + (BH_COMPLEX64 << 4)
        case 173: return true; // Cf: BH_COMPLEX128 + (BH_FLOAT32 << 4)
        case 189: return true; // Cd: BH_COMPLEX128 + (BH_FLOAT64 << 4)
        case 61: return true; // Cs: BH_COMPLEX128 + (BH_INT16 << 4)
        case 77: return true; // Ci: BH_COMPLEX128 + (BH_INT32 << 4)
        case 93: return true; // Cl: BH_COMPLEX128 + (BH_INT64 << 4)
        case 45: return true; // Cb: BH_COMPLEX128 + (BH_INT8 << 4)
        case 125: return true; // CS: BH_COMPLEX128 + (BH_UINT16 << 4)
        case 141: return true; // CI: BH_COMPLEX128 + (BH_UINT32 << 4)
        case 157: return true; // CL: BH_COMPLEX128 + (BH_UINT64 << 4)
        case 109: return true; // CB: BH_COMPLEX128 + (BH_UINT8 << 4)
        case 28: return true; // cz: BH_COMPLEX64 + (BH_BOOL << 4)
        case 220: return true; // cC: BH_COMPLEX64 + (BH_COMPLEX128 << 4)
        case 172: return true; // cf: BH_COMPLEX64 + (BH_FLOAT32 << 4)
        case 188: return true; // cd: BH_COMPLEX64 + (BH_FLOAT64 << 4)
        case 60: return true; // cs: BH_COMPLEX64 + (BH_INT16 << 4)
        case 76: return true; // ci: BH_COMPLEX64 + (BH_INT32 << 4)
        case 92: return true; // cl: BH_COMPLEX64 + (BH_INT64 << 4)
        case 44: return true; // cb: BH_COMPLEX64 + (BH_INT8 << 4)
        case 124: return true; // cS: BH_COMPLEX64 + (BH_UINT16 << 4)
        case 140: return true; // cI: BH_COMPLEX64 + (BH_UINT32 << 4)
        case 156: return true; // cL: BH_COMPLEX64 + (BH_UINT64 << 4)
        case 108: return true; // cB: BH_COMPLEX64 + (BH_UINT8 << 4)
        case 26: return true; // fz: BH_FLOAT32 + (BH_BOOL << 4)
        case 186: return true; // fd: BH_FLOAT32 + (BH_FLOAT64 << 4)
        case 58: return true; // fs: BH_FLOAT32 + (BH_INT16 << 4)
        case 74: return true; // fi: BH_FLOAT32 + (BH_INT32 << 4)
        case 90: return true; // fl: BH_FLOAT32 + (BH_INT64 << 4)
        case 42: return true; // fb: BH_FLOAT32 + (BH_INT8 << 4)
        case 122: return true; // fS: BH_FLOAT32 + (BH_UINT16 << 4)
        case 138: return true; // fI: BH_FLOAT32 + (BH_UINT32 << 4)
        case 154: return true; // fL: BH_FLOAT32 + (BH_UINT64 << 4)
        case 106: return true; // fB: BH_FLOAT32 + (BH_UINT8 << 4)
        case 27: return true; // dz: BH_FLOAT64 + (BH_BOOL << 4)
        case 171: return true; // df: BH_FLOAT64 + (BH_FLOAT32 << 4)
        case 59: return true; // ds: BH_FLOAT64 + (BH_INT16 << 4)
        case 75: return true; // di: BH_FLOAT64 + (BH_INT32 << 4)
        case 91: return true; // dl: BH_FLOAT64 + (BH_INT64 << 4)
        case 43: return true; // db: BH_FLOAT64 + (BH_INT8 << 4)
        case 123: return true; // dS: BH_FLOAT64 + (BH_UINT16 << 4)
        case 139: return true; // dI: BH_FLOAT64 + (BH_UINT32 << 4)
        case 155: return true; // dL: BH_FLOAT64 + (BH_UINT64 << 4)
        case 107: return true; // dB: BH_FLOAT64 + (BH_UINT8 << 4)
        case 19: return true; // sz: BH_INT16 + (BH_BOOL << 4)
        case 163: return true; // sf: BH_INT16 + (BH_FLOAT32 << 4)
        case 179: return true; // sd: BH_INT16 + (BH_FLOAT64 << 4)
        case 67: return true; // si: BH_INT16 + (BH_INT32 << 4)
        case 83: return true; // sl: BH_INT16 + (BH_INT64 << 4)
        case 35: return true; // sb: BH_INT16 + (BH_INT8 << 4)
        case 115: return true; // sS: BH_INT16 + (BH_UINT16 << 4)
        case 131: return true; // sI: BH_INT16 + (BH_UINT32 << 4)
        case 147: return true; // sL: BH_INT16 + (BH_UINT64 << 4)
        case 99: return true; // sB: BH_INT16 + (BH_UINT8 << 4)
        case 20: return true; // iz: BH_INT32 + (BH_BOOL << 4)
        case 164: return true; // if: BH_INT32 + (BH_FLOAT32 << 4)
        case 180: return true; // id: BH_INT32 + (BH_FLOAT64 << 4)
        case 52: return true; // is: BH_INT32 + (BH_INT16 << 4)
        case 84: return true; // il: BH_INT32 + (BH_INT64 << 4)
        case 36: return true; // ib: BH_INT32 + (BH_INT8 << 4)
        case 116: return true; // iS: BH_INT32 + (BH_UINT16 << 4)
        case 132: return true; // iI: BH_INT32 + (BH_UINT32 << 4)
        case 148: return true; // iL: BH_INT32 + (BH_UINT64 << 4)
        case 100: return true; // iB: BH_INT32 + (BH_UINT8 << 4)
        case 21: return true; // lz: BH_INT64 + (BH_BOOL << 4)
        case 165: return true; // lf: BH_INT64 + (BH_FLOAT32 << 4)
        case 181: return true; // ld: BH_INT64 + (BH_FLOAT64 << 4)
        case 53: return true; // ls: BH_INT64 + (BH_INT16 << 4)
        case 69: return true; // li: BH_INT64 + (BH_INT32 << 4)
        case 37: return true; // lb: BH_INT64 + (BH_INT8 << 4)
        case 117: return true; // lS: BH_INT64 + (BH_UINT16 << 4)
        case 133: return true; // lI: BH_INT64 + (BH_UINT32 << 4)
        case 149: return true; // lL: BH_INT64 + (BH_UINT64 << 4)
        case 101: return true; // lB: BH_INT64 + (BH_UINT8 << 4)
        case 18: return true; // bz: BH_INT8 + (BH_BOOL << 4)
        case 162: return true; // bf: BH_INT8 + (BH_FLOAT32 << 4)
        case 178: return true; // bd: BH_INT8 + (BH_FLOAT64 << 4)
        case 50: return true; // bs: BH_INT8 + (BH_INT16 << 4)
        case 66: return true; // bi: BH_INT8 + (BH_INT32 << 4)
        case 82: return true; // bl: BH_INT8 + (BH_INT64 << 4)
        case 114: return true; // bS: BH_INT8 + (BH_UINT16 << 4)
        case 130: return true; // bI: BH_INT8 + (BH_UINT32 << 4)
        case 146: return true; // bL: BH_INT8 + (BH_UINT64 << 4)
        case 98: return true; // bB: BH_INT8 + (BH_UINT8 << 4)
        case 23: return true; // Sz: BH_UINT16 + (BH_BOOL << 4)
        case 167: return true; // Sf: BH_UINT16 + (BH_FLOAT32 << 4)
        case 183: return true; // Sd: BH_UINT16 + (BH_FLOAT64 << 4)
        case 55: return true; // Ss: BH_UINT16 + (BH_INT16 << 4)
        case 71: return true; // Si: BH_UINT16 + (BH_INT32 << 4)
        case 87: return true; // Sl: BH_UINT16 + (BH_INT64 << 4)
        case 39: return true; // Sb: BH_UINT16 + (BH_INT8 << 4)
        case 135: return true; // SI: BH_UINT16 + (BH_UINT32 << 4)
        case 151: return true; // SL: BH_UINT16 + (BH_UINT64 << 4)
        case 103: return true; // SB: BH_UINT16 + (BH_UINT8 << 4)
        case 24: return true; // Iz: BH_UINT32 + (BH_BOOL << 4)
        case 168: return true; // If: BH_UINT32 + (BH_FLOAT32 << 4)
        case 184: return true; // Id: BH_UINT32 + (BH_FLOAT64 << 4)
        case 56: return true; // Is: BH_UINT32 + (BH_INT16 << 4)
        case 72: return true; // Ii: BH_UINT32 + (BH_INT32 << 4)
        case 88: return true; // Il: BH_UINT32 + (BH_INT64 << 4)
        case 40: return true; // Ib: BH_UINT32 + (BH_INT8 << 4)
        case 120: return true; // IS: BH_UINT32 + (BH_UINT16 << 4)
        case 152: return true; // IL: BH_UINT32 + (BH_UINT64 << 4)
        case 104: return true; // IB: BH_UINT32 + (BH_UINT8 << 4)
        case 25: return true; // Lz: BH_UINT64 + (BH_BOOL << 4)
        case 169: return true; // Lf: BH_UINT64 + (BH_FLOAT32 << 4)
        case 185: return true; // Ld: BH_UINT64 + (BH_FLOAT64 << 4)
        case 57: return true; // Ls: BH_UINT64 + (BH_INT16 << 4)
        case 73: return true; // Li: BH_UINT64 + (BH_INT32 << 4)
        case 89: return true; // Ll: BH_UINT64 + (BH_INT64 << 4)
        case 41: return true; // Lb: BH_UINT64 + (BH_INT8 << 4)
        case 121: return true; // LS: BH_UINT64 + (BH_UINT16 << 4)
        case 137: return true; // LI: BH_UINT64 + (BH_UINT32 << 4)
        case 105: return true; // LB: BH_UINT64 + (BH_UINT8 << 4)
        case 22: return true; // Bz: BH_UINT8 + (BH_BOOL << 4)
        case 166: return true; // Bf: BH_UINT8 + (BH_FLOAT32 << 4)
        case 182: return true; // Bd: BH_UINT8 + (BH_FLOAT64 << 4)
        case 54: return true; // Bs: BH_UINT8 + (BH_INT16 << 4)
        case 70: return true; // Bi: BH_UINT8 + (BH_INT32 << 4)
        case 86: return true; // Bl: BH_UINT8 + (BH_INT64 << 4)
        case 38: return true; // Bb: BH_UINT8 + (BH_INT8 << 4)
        case 118: return true; // BS: BH_UINT8 + (BH_UINT16 << 4)
        case 134: return true; // BI: BH_UINT8 + (BH_UINT32 << 4)
        case 150: return true; // BL: BH_UINT8 + (BH_UINT64 << 4)
        case 232: return true; // IR: BH_UINT32 + (BH_R123 << 4)
        case 233: return true; // LR: BH_UINT64 + (BH_R123 << 4)
        case 219: return true; // dC: BH_FLOAT64 + (BH_COMPLEX128 << 4)
        case 202: return true; // fc: BH_FLOAT32 + (BH_COMPLEX64 << 4)
        case 8: return true; // I: BH_UINT32
        case 9: return true; // L: BH_UINT64

        default:
            return false;
    }
}

/* Determines if the types are acceptable for the operation
 *
 * @opcode Opcode for operation
 * @outtype The type of the output
 * @inputtype1 The type of the first input operand
 * @inputtype2 The type of the second input operand
 * @constanttype The type of the constant
 * @return TRUE if the types are accepted, FALSE otherwise
 */
bool bh_validate_types(bh_opcode opcode, bh_type outtype, bh_type inputtype1, bh_type inputtype2, bh_type constanttype)
{
    // Poly contains a unique value, pairing an opcode with its function signature.
    // All in one nicely switchable value.
    long int poly;
 
    if (bh_operands(opcode) == 3) {                 // Three operands

        if (inputtype1 == BH_UNKNOWN) {             // First operand is constant
            poly = opcode \
                | (outtype << 8) \
                | (constanttype << 12) \
                | (inputtype2 << 16);

        } else if (inputtype2 == BH_UNKNOWN) {      // Second operand is constant
            poly = opcode
                | (outtype << 8) \
                | (inputtype1 << 12) \
                | (constanttype << 16);

        } else {                                     // No constant operand
            poly = opcode \
                | (outtype << 8) \
                | (inputtype1 << 12) \
                | (inputtype2 << 16);
        }

    } else {                                         // Two operands

        if (inputtype1 == BH_UNKNOWN) {
            poly = opcode \
                | (outtype << 8) \
                | (constanttype << 12) \
                | (1 << 17);
        } else {
            poly = opcode \
                | (outtype << 8) \
                | (inputtype1 << 12) \
                | (1 << 17);
        }
    }
    

    switch (poly)
    {
        case BH_ADD | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_ADD | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_ADD | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_ADD | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_ADD | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_ADD | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_ADD | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_ADD | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_ADD | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_ADD | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_ADD | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_ADD | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_ADD | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_SUBTRACT | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_SUBTRACT | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_SUBTRACT | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_SUBTRACT | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_SUBTRACT | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_SUBTRACT | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_SUBTRACT | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_SUBTRACT | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_SUBTRACT | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_SUBTRACT | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_SUBTRACT | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_SUBTRACT | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_SUBTRACT | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_MULTIPLY | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_MULTIPLY | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_MULTIPLY | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_MULTIPLY | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_MULTIPLY | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_MULTIPLY | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_MULTIPLY | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_MULTIPLY | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_MULTIPLY | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_MULTIPLY | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_MULTIPLY | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_MULTIPLY | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_MULTIPLY | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_DIVIDE | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_DIVIDE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_DIVIDE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_DIVIDE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_DIVIDE | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_DIVIDE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_DIVIDE | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_DIVIDE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_DIVIDE | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_DIVIDE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_DIVIDE | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_DIVIDE | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_POWER | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_POWER | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_POWER | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_POWER | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_POWER | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_POWER | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_POWER | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_POWER | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_POWER | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_POWER | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_ABSOLUTE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_ABSOLUTE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_GREATER | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_GREATER | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_GREATER_EQUAL | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_LESS | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_LESS_EQUAL | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_EQUAL | (BH_BOOL << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_COMPLEX64 << 12) | (BH_COMPLEX64 << 16):
        case BH_NOT_EQUAL | (BH_BOOL << 8) | (BH_COMPLEX128 << 12) | (BH_COMPLEX128 << 16):
        case BH_LOGICAL_AND | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_LOGICAL_OR | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_LOGICAL_XOR | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_LOGICAL_NOT | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_MAXIMUM | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_MAXIMUM | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_MAXIMUM | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_MAXIMUM | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_MAXIMUM | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_MAXIMUM | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_MAXIMUM | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_MAXIMUM | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_MAXIMUM | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_MAXIMUM | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_MAXIMUM | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_MINIMUM | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_MINIMUM | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_MINIMUM | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_MINIMUM | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_MINIMUM | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_MINIMUM | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_MINIMUM | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_MINIMUM | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_MINIMUM | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_MINIMUM | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_MINIMUM | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_BITWISE_AND | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_BITWISE_AND | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_BITWISE_AND | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_BITWISE_AND | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_BITWISE_AND | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_BITWISE_AND | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_BITWISE_AND | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_BITWISE_AND | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_BITWISE_AND | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_BITWISE_OR | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_BITWISE_OR | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_BITWISE_OR | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_BITWISE_OR | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_BITWISE_OR | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_BITWISE_OR | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_BITWISE_OR | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_BITWISE_OR | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_BITWISE_OR | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_BITWISE_XOR | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_BITWISE_XOR | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_BITWISE_XOR | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_BITWISE_XOR | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_BITWISE_XOR | (BH_BOOL << 8) | (BH_BOOL << 12) | (BH_BOOL << 16):
        case BH_BITWISE_XOR | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_BITWISE_XOR | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_BITWISE_XOR | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_BITWISE_XOR | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_INVERT | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_INVERT | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_INVERT | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_INVERT | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_INVERT | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_INVERT | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_INVERT | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_INVERT | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_INVERT | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_LEFT_SHIFT | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_LEFT_SHIFT | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_LEFT_SHIFT | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_LEFT_SHIFT | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_LEFT_SHIFT | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_LEFT_SHIFT | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_LEFT_SHIFT | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_LEFT_SHIFT | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_RIGHT_SHIFT | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_RIGHT_SHIFT | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_RIGHT_SHIFT | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_RIGHT_SHIFT | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_RIGHT_SHIFT | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_RIGHT_SHIFT | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_RIGHT_SHIFT | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_RIGHT_SHIFT | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_COS | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_COS | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_COS | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_COS | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_SIN | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_SIN | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_SIN | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_SIN | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_TAN | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_TAN | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_TAN | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_TAN | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_COSH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_COSH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_COSH | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_COSH | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_SINH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_SINH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_SINH | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_SINH | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_TANH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_TANH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_TANH | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_TANH | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_ARCSIN | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCSIN | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCCOS | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCCOS | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCTAN | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCTAN | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCSINH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCSINH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCCOSH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCCOSH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCTANH | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ARCTANH | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ARCTAN2 | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_ARCTAN2 | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_EXP | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_EXP | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_EXP | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_EXP | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_EXP2 | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_EXP2 | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_EXPM1 | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_EXPM1 | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_LOG | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_LOG | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_LOG | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_LOG | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_LOG2 | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_LOG2 | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_LOG10 | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_LOG10 | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_LOG10 | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_LOG10 | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_LOG1P | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_LOG1P | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_SQRT | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_SQRT | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_SQRT | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_SQRT | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_CEIL | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_CEIL | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_TRUNC | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_TRUNC | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_FLOOR | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_FLOOR | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_RINT | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_RINT | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_MOD | (BH_INT8 << 8) | (BH_INT8 << 12) | (BH_INT8 << 16):
        case BH_MOD | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (BH_FLOAT64 << 16):
        case BH_MOD | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (BH_UINT16 << 16):
        case BH_MOD | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (BH_UINT64 << 16):
        case BH_MOD | (BH_INT16 << 8) | (BH_INT16 << 12) | (BH_INT16 << 16):
        case BH_MOD | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (BH_FLOAT32 << 16):
        case BH_MOD | (BH_INT32 << 8) | (BH_INT32 << 12) | (BH_INT32 << 16):
        case BH_MOD | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (BH_UINT8 << 16):
        case BH_MOD | (BH_INT64 << 8) | (BH_INT64 << 12) | (BH_INT64 << 16):
        case BH_MOD | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (BH_UINT32 << 16):
        case BH_ISNAN | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ISNAN | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ISINF | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ISINF | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_BOOL << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT8 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT8 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT16 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT16 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_INT64 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_UINT64 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_FLOAT64 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX128 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_IDENTITY | (BH_COMPLEX64 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_ADD_REDUCE | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_COMPLEX64 << 8) | (BH_COMPLEX64 << 12) | (1 << 17):
        case BH_MULTIPLY_REDUCE | (BH_COMPLEX128 << 8) | (BH_COMPLEX128 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_MINIMUM_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_FLOAT64 << 8) | (BH_FLOAT64 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_FLOAT32 << 8) | (BH_FLOAT32 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_MAXIMUM_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_LOGICAL_AND_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_BITWISE_AND_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_LOGICAL_OR_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_BITWISE_OR_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
        case BH_LOGICAL_XOR_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_INT8 << 8) | (BH_INT8 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_UINT16 << 8) | (BH_UINT16 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_UINT64 << 8) | (BH_UINT64 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_INT16 << 8) | (BH_INT16 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_BOOL << 8) | (BH_BOOL << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_INT32 << 8) | (BH_INT32 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_UINT8 << 8) | (BH_UINT8 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_INT64 << 8) | (BH_INT64 << 12) | (1 << 17):
        case BH_BITWISE_XOR_REDUCE | (BH_UINT32 << 8) | (BH_UINT32 << 12) | (1 << 17):
            return true;
                    
        default:
            return false;
    }

}

/* Determines if the types are acceptable for the operation, 
 * and provides a suggestion for converting them
 *
 * @opcode Opcode for operation
 * @outtype The type of the output
 * @inputtype1 The type of the first input operand
 * @inputtype2 The type of the second input operand
 * @constanttype The type of the constant
 * @return TRUE if the types can be converted, FALSE otherwise
 */
bool bh_get_type_conversion(bh_opcode opcode, bh_type outtype, bh_type* inputtype1, bh_type* inputtype2, bh_type* constanttype) 
{
    // Basic case, "it just works"
    if (bh_validate_types(opcode, outtype, *inputtype1, *inputtype2, *constanttype))
        return true;
        
    // All valid identity types are covered
    if (opcode == BH_IDENTITY)
        return false;
    
    // Grab the input types
    bh_type desired_input_type1 = BH_UNKNOWN;
    bh_type desired_input_type2 = BH_UNKNOWN;

    // Poly contains a unique value, pairing an opcode with its function signature.
    // All in one nicely switchable value.
    long int poly;
    
    poly = opcode | (outtype << 8);

    switch(poly)
    {
            case BH_ADD | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_ADD | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_ADD | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_ADD | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_ADD | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_ADD | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_ADD | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_ADD | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_ADD | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_ADD | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_ADD | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_ADD | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                desired_input_type2 = BH_COMPLEX64;
                break;
            case BH_ADD | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                desired_input_type2 = BH_COMPLEX128;
                break;
            case BH_SUBTRACT | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_SUBTRACT | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_SUBTRACT | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_SUBTRACT | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_SUBTRACT | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_SUBTRACT | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_SUBTRACT | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_SUBTRACT | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_SUBTRACT | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_SUBTRACT | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_SUBTRACT | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_SUBTRACT | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                desired_input_type2 = BH_COMPLEX64;
                break;
            case BH_SUBTRACT | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                desired_input_type2 = BH_COMPLEX128;
                break;
            case BH_MULTIPLY | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_MULTIPLY | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_MULTIPLY | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_MULTIPLY | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_MULTIPLY | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_MULTIPLY | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_MULTIPLY | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_MULTIPLY | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_MULTIPLY | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_MULTIPLY | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_MULTIPLY | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_MULTIPLY | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                desired_input_type2 = BH_COMPLEX64;
                break;
            case BH_MULTIPLY | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                desired_input_type2 = BH_COMPLEX128;
                break;
            case BH_DIVIDE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_DIVIDE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_DIVIDE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_DIVIDE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_DIVIDE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_DIVIDE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_DIVIDE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_DIVIDE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_DIVIDE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_DIVIDE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_DIVIDE | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                desired_input_type2 = BH_COMPLEX64;
                break;
            case BH_DIVIDE | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                desired_input_type2 = BH_COMPLEX128;
                break;
            case BH_POWER | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_POWER | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_POWER | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_POWER | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_POWER | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_POWER | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_POWER | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_POWER | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_POWER | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_POWER | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_ABSOLUTE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_ABSOLUTE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ABSOLUTE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_ABSOLUTE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_ABSOLUTE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_ABSOLUTE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ABSOLUTE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_ABSOLUTE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_ABSOLUTE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_ABSOLUTE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_ABSOLUTE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_GREATER | (BH_BOOL << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_MAXIMUM | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_MAXIMUM | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_MAXIMUM | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_MAXIMUM | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_MAXIMUM | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_MAXIMUM | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_MAXIMUM | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_MAXIMUM | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_MAXIMUM | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_MAXIMUM | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_MAXIMUM | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_MINIMUM | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_MINIMUM | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_MINIMUM | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_MINIMUM | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_MINIMUM | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_MINIMUM | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_MINIMUM | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_MINIMUM | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_MINIMUM | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_MINIMUM | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_MINIMUM | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_BITWISE_AND | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_BITWISE_AND | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_BITWISE_AND | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_BITWISE_AND | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_BITWISE_AND | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_BITWISE_AND | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_BITWISE_AND | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_BITWISE_AND | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_BITWISE_AND | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_BITWISE_OR | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_BITWISE_OR | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_BITWISE_OR | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_BITWISE_OR | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_BITWISE_OR | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_BITWISE_OR | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_BITWISE_OR | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_BITWISE_OR | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_BITWISE_OR | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_BITWISE_XOR | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_BITWISE_XOR | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_BITWISE_XOR | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_BITWISE_XOR | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_BITWISE_XOR | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                desired_input_type2 = BH_BOOL;
                break;
            case BH_BITWISE_XOR | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_BITWISE_XOR | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_BITWISE_XOR | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_BITWISE_XOR | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_INVERT | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_INVERT | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_INVERT | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_INVERT | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_INVERT | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_INVERT | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_INVERT | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_INVERT | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_INVERT | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_LEFT_SHIFT | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_LEFT_SHIFT | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_LEFT_SHIFT | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_LEFT_SHIFT | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_LEFT_SHIFT | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_LEFT_SHIFT | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_LEFT_SHIFT | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_LEFT_SHIFT | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_RIGHT_SHIFT | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_RIGHT_SHIFT | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_RIGHT_SHIFT | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_RIGHT_SHIFT | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_RIGHT_SHIFT | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_RIGHT_SHIFT | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_RIGHT_SHIFT | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_RIGHT_SHIFT | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_COS | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_COS | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_COS | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_COS | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_SIN | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_SIN | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_SIN | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_SIN | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_TAN | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_TAN | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_TAN | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_TAN | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_COSH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_COSH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_COSH | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_COSH | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_SINH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_SINH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_SINH | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_SINH | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_TANH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_TANH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_TANH | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_TANH | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_ARCSIN | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCSIN | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCCOS | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCCOS | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCTAN | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCTAN | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCSINH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCSINH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCCOSH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCCOSH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCTANH | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ARCTANH | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ARCTAN2 | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_ARCTAN2 | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_EXP | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_EXP | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_EXP | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_EXP | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_EXP2 | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_EXP2 | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_EXPM1 | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_EXPM1 | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_LOG | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_LOG | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_LOG | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_LOG | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_LOG2 | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_LOG2 | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_LOG10 | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_LOG10 | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_LOG10 | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_LOG10 | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_LOG1P | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_LOG1P | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_SQRT | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_SQRT | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_SQRT | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_SQRT | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_CEIL | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_CEIL | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_TRUNC | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_TRUNC | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_FLOOR | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_FLOOR | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_RINT | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_RINT | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_MOD | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                desired_input_type2 = BH_INT8;
                break;
            case BH_MOD | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                desired_input_type2 = BH_FLOAT64;
                break;
            case BH_MOD | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                desired_input_type2 = BH_UINT16;
                break;
            case BH_MOD | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                desired_input_type2 = BH_UINT64;
                break;
            case BH_MOD | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                desired_input_type2 = BH_INT16;
                break;
            case BH_MOD | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                desired_input_type2 = BH_FLOAT32;
                break;
            case BH_MOD | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                desired_input_type2 = BH_INT32;
                break;
            case BH_MOD | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                desired_input_type2 = BH_UINT8;
                break;
            case BH_MOD | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                desired_input_type2 = BH_INT64;
                break;
            case BH_MOD | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                desired_input_type2 = BH_UINT32;
                break;
            case BH_ISNAN | (BH_BOOL << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ADD_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_ADD_REDUCE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_ADD_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_ADD_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_ADD_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_ADD_REDUCE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_ADD_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_ADD_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_ADD_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_ADD_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_ADD_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_ADD_REDUCE | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_ADD_REDUCE | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_MULTIPLY_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_MULTIPLY_REDUCE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_MULTIPLY_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_MULTIPLY_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_MULTIPLY_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_MULTIPLY_REDUCE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_MULTIPLY_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_MULTIPLY_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_MULTIPLY_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_MULTIPLY_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_MULTIPLY_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_MULTIPLY_REDUCE | (BH_COMPLEX64 << 8):
                desired_input_type1 = BH_COMPLEX64;
                break;
            case BH_MULTIPLY_REDUCE | (BH_COMPLEX128 << 8):
                desired_input_type1 = BH_COMPLEX128;
                break;
            case BH_MINIMUM_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_MINIMUM_REDUCE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_MINIMUM_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_MINIMUM_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_MINIMUM_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_MINIMUM_REDUCE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_MINIMUM_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_MINIMUM_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_MINIMUM_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_MINIMUM_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_MINIMUM_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_MAXIMUM_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_MAXIMUM_REDUCE | (BH_FLOAT64 << 8):
                desired_input_type1 = BH_FLOAT64;
                break;
            case BH_MAXIMUM_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_MAXIMUM_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_MAXIMUM_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_MAXIMUM_REDUCE | (BH_FLOAT32 << 8):
                desired_input_type1 = BH_FLOAT32;
                break;
            case BH_MAXIMUM_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_MAXIMUM_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_MAXIMUM_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_MAXIMUM_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_MAXIMUM_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_LOGICAL_AND_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_BITWISE_AND_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_LOGICAL_OR_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_BITWISE_OR_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
            case BH_LOGICAL_XOR_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_INT8 << 8):
                desired_input_type1 = BH_INT8;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_UINT16 << 8):
                desired_input_type1 = BH_UINT16;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_UINT64 << 8):
                desired_input_type1 = BH_UINT64;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_INT16 << 8):
                desired_input_type1 = BH_INT16;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_BOOL << 8):
                desired_input_type1 = BH_BOOL;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_INT32 << 8):
                desired_input_type1 = BH_INT32;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_UINT8 << 8):
                desired_input_type1 = BH_UINT8;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_INT64 << 8):
                desired_input_type1 = BH_INT64;
                break;
            case BH_BITWISE_XOR_REDUCE | (BH_UINT32 << 8):
                desired_input_type1 = BH_UINT32;
                break;
    }
    
    // The output type does not exist for the opcode
    if (desired_input_type1 == BH_UNKNOWN)
        return false;
    
    if (bh_operands(opcode) == 3)
    {
        // The output type does not exist for the opcode
        if (desired_input_type2 == BH_UNKNOWN)
            return false;

        if (*inputtype1 == BH_UNKNOWN)                      // First operand constant
        {
        // Check if we can convert with IDENTITY
            if (bh_validate_types(BH_IDENTITY, desired_input_type1, *constanttype, BH_UNKNOWN, BH_UNKNOWN) && bh_validate_types(BH_IDENTITY, desired_input_type2, *inputtype2, BH_UNKNOWN, BH_UNKNOWN))
            {
                *constanttype = desired_input_type1;
                *inputtype2 = desired_input_type2;
                return true;
            }
        }
        else if (*inputtype2 == BH_UNKNOWN)                 // Second operand constant
        {
            // Check if we can convert with IDENTITY
            if (bh_validate_types(BH_IDENTITY, desired_input_type1, *inputtype1, BH_UNKNOWN, BH_UNKNOWN) && bh_validate_types(BH_IDENTITY, desired_input_type2, *constanttype, BH_UNKNOWN, BH_UNKNOWN))
            {
                *inputtype1 = desired_input_type1;
                *constanttype = desired_input_type2;
                return true;
            }
        }
        else                                                // No constant
        {
            // Check if we can convert with IDENTITY
            if (bh_validate_types(BH_IDENTITY, desired_input_type1, *inputtype1, BH_UNKNOWN, BH_UNKNOWN) && bh_validate_types(BH_IDENTITY, desired_input_type2, *inputtype2, BH_UNKNOWN, BH_UNKNOWN))
            {
                *inputtype1 = desired_input_type1;
                *inputtype2 = desired_input_type2;
                return true;
            }
        }
    }
    else
    {
        // Check if we can convert with IDENTITY
        if (*inputtype1 == BH_UNKNOWN)                      // Constant input
        {
            if (bh_validate_types(BH_IDENTITY, desired_input_type1, *constanttype, BH_UNKNOWN, BH_UNKNOWN))
            {
                *constanttype = desired_input_type1;
                return true;
            }
        }
        else                                                // No constant
        {
            if (bh_validate_types(BH_IDENTITY, desired_input_type1, *inputtype1, BH_UNKNOWN, BH_UNKNOWN))
            {
                *inputtype1 = desired_input_type1;
                return true;
            }
        }
    }
    
    // Not possible to convert the types automatically
    return false;
}
