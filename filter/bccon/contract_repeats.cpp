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
#include "contracter.hpp"

#include <boost/regex.hpp>
#include <iostream>
#include <string>
#include <unordered_map>

#include <bh_component.hpp>

using namespace std;

namespace bohrium {
namespace filter {
namespace bccon {

// Regex matcher. This regex will match all repeating groups in a string
// ie. will match 'bc' in 'abcbcbcde'
static const boost::regex re("(.+)\\1+");

// Convert the instruction list to a string of characters
// Identical instructions will be mapped to the same character
static string bh_instr_list_to_string(const vector<bh_instruction> &instr_list, unordered_map<char, const bh_instruction*> &identifier_map)
{
    string result = "";
    char identifier = (char)((int) 'a' - 1);

    for(const bh_instruction &instr : instr_list) {
        bool seen = false;

        // See if we have already seen this instruction before
        for (const auto it : identifier_map) {
            if (*it.second == instr) {
                identifier = it.first;
                seen = true;
                break;
            }
        }

        // If we haven't seen the instruction before, update the identifier and add it to the map
        if (!seen) {
            identifier = (char)((int) identifier + 1);
            identifier_map[identifier] = &instr;
        }

        result += identifier;
    }

    return result;
}

// Count occurrences of str2 in str1
static uint count_occur(string str1, string str2)
{
    uint occur = 0;
    uint start = 0;

    while ((start = str1.find(str2, start)) < str1.length()) {
        ++occur;
        start += str2.length();
    }

    return occur;
}

void Contracter::contract_repeats(BhIR &bhir)
{
    // Build map of instructions and string representation of instruction list
    unordered_map<char, const bh_instruction*> identifier_map;
    const string bh_string_instr_list = bh_instr_list_to_string(bhir.instr_list, identifier_map);

    boost::smatch matches;

    try {
        if (boost::regex_search(bh_string_instr_list.begin(), bh_string_instr_list.end(), matches, re)) {
            uint size  = matches.str(1).size(); // How many instructions
            uint occur = count_occur(matches.str(0), matches.str(1)); // How many repeats

            verbose_print("[Repeat] Found repeating sequence of length " + std::to_string(occur) + ". It repeats " + std::to_string(size) + " times.");

            // Replace the matches with 'R'
            string instr_list_str;
            boost::regex_replace(back_inserter(instr_list_str), bh_string_instr_list.begin(), bh_string_instr_list.end(), re, "R");

            vector<bh_instruction> new_bh_instr_list;

            for(const char c : instr_list_str) {
                // A 'R' means we have found our repeating sequence
                if (c == 'R') {
                    // Create a BH_REPEAT instruction
                    bh_instruction repeat_instr;
                    repeat_instr.opcode = BH_REPEAT;
                    repeat_instr.operand[0].base = NULL;

                    // We use the BH_R123 type, because this can hold two values
                    bh_constant constant;
                    constant.type = bh_type::R123;
                    constant.value.r123.start = size;
                    constant.value.r123.key = occur;
                    repeat_instr.constant = constant;

                    // BH_REPEAT is pushed to new instruction list
                    new_bh_instr_list.push_back(repeat_instr);

                    // Push inner part of the repeat to new instruction list
                    for(const char d : matches.str(1)) {
                        new_bh_instr_list.push_back(*identifier_map.find(d)->second);
                    }
                } else {
                    // This is a regular instruction. Push as is.
                    new_bh_instr_list.push_back(*identifier_map.find(c)->second);
                }
            }

            // Set new instruction list
            bhir.instr_list = new_bh_instr_list;
        }
    } catch(std::runtime_error& e) {
        verbose_print("[Repeat] \tRegex failed - Moving on");
        return;
    }

    return;
}

}}}
