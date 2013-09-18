#include <string>
#include <vector>
#include <stdexcept>
#include <ctemplate/template.h>  
#include "compiler.cpp"
#include <jansson.h>
#include "bh.h"

using namespace std;

string generate_source(
    string template_fn, string symbol, string cexpr,
    string type_out, string type_in1, string type_in2,
    string structure, bool license, bool include
    )
{
    string sourcecode = ""; 
    ctemplate::TemplateDictionary dict(symbol);
    dict.SetValue("OPERATOR",   cexpr);
    dict.SetValue("SYMBOL",     symbol);
    dict.SetValue("TYPE_A0",    type_out);
    dict.SetValue("TYPE_A1",    type_in1);
    if (""!=type_in2) {
        dict.SetValue("TYPE_A2", type_in2);
    }
    if (structure=="DDD") {
        dict.ShowSection("a1_dense");
        dict.ShowSection("a2_dense");
    } else if(structure=="DDC") {
        dict.ShowSection("a1_dense");
        dict.ShowSection("a2_scalar");
    } else if(structure=="DCD") {
        dict.ShowSection("a1_scalar");
        dict.ShowSection("a2_dense");
    } else if(structure=="DD") {
        dict.ShowSection("a1_dense");
    } else if(structure=="DC") {
        dict.ShowSection("a1_scalar");
    }
    if (license) {
        dict.ShowSection("license");
    }
    if (include) {
        dict.ShowSection("include");
    }
    ctemplate::ExpandTemplate(
        template_fn,
        ctemplate::STRIP_BLANK_LINES,
        &dict,
        &sourcecode
    );
    return sourcecode;
}

void precompile(
    const char *cmd,  const char *jsonfile,
    const char *object_path, const char *kernel_path, const char *template_path,
    bool scattered)
{
    //process target(cmd, object_path, kernel_path, false);
    process target(cmd, object_path, kernel_path, scattered);
    string op,
            symbol,
            type_out,
            type_in1,
            type_in2,
            template_fn,
            sourcecode;

    bool license = true;
    bool include = true;

    json_t *root;
    json_error_t error;
    const char *dimensions[]        = {"1d", "2d", "3d", "naive"};
    const char *binary_structs[]    = {"DDD",   "DDC", "DCD"};
    const char *unary_structs[]     = {"DD",    "DC"};

    std::vector<std::string> symbol_table;    // List of symbols compiled.

    root = json_load_file(                              // Basis for pre-compilation
        jsonfile,
        0,
        &error
    );
    if (!json_is_array(root)) {                         // Load the opcodes
        fprintf(stderr, "error: root is not an array\n");
    }

    for (size_t i=0; i < json_array_size(root); ++i) {  // Run through them all
        json_t *opcode_j, *signatures_j, *signature_j;
        
        opcode_j = json_array_get(root, i);
        if(!json_is_object(opcode_j)) {
            fprintf(stderr, "error: commit data %lu is not an object\n", i + 1);
        }
                                                    
        const char *opcode  = json_string_value(json_object_get(opcode_j,   "opcode"));
        size_t id           = json_integer_value(json_object_get(opcode_j,  "id"));
        size_t nop          = json_integer_value(json_object_get(opcode_j,  "nop"));
        int system_opcode   = json_is_true(json_object_get(opcode_j,        "system_opcode"));
        int elementwise     = json_is_true(json_object_get(opcode_j,        "elementwise"));
        const char  *type_out = "",
                    *type_in1 = "",
                    *type_in2 = "",
                    *structure= "";

        signatures_j = json_object_get(opcode_j, "types");
        for(size_t t=0; t<json_array_size(signatures_j); ++t) {
            signature_j = json_array_get(signatures_j, t);
            std::string signature = "";
            for(size_t s=0; s<json_array_size(signature_j); ++s) {
                if (s==0) {
                    type_out = json_string_value(json_array_get(signature_j, s));
                    signature += std::string(bhtypestr_to_shorthand(type_out));
                    type_out = typestr_to_ctype(type_out);
                } else if (s==1) {
                    type_in1 = json_string_value(json_array_get(signature_j, s));
                    signature += std::string(bhtypestr_to_shorthand(type_in1));
                    type_in1 = typestr_to_ctype(type_in1);
                } else if (s==2) {
                    type_in2 = json_string_value(json_array_get(signature_j, s));
                    signature += std::string(bhtypestr_to_shorthand(type_in2));
                    type_in2 = typestr_to_ctype(type_in2);
                }
            }
            if (json_array_size(signature_j) < 3) {
                type_in2 = "";
            }
            if ((elementwise) && (!system_opcode)) {            // TRAVERSE
                for(size_t d=0; d<4; ++d) {             
                    for(size_t s=0; s<nop;++s) {
                        symbol = opcode;
                        symbol += "_"+ string(dimensions[d]) + "_";
                        if (nop==3) {
                            structure = binary_structs[s];
                        } else if (nop==2) {
                            structure = unary_structs[s];
                        }
                        symbol += string(structure);
                        symbol += "_";
                        symbol += string(signature);

                        if (!target.symbol_ready(symbol)) {
                            template_fn = string(template_path) + "/traverse."+dimensions[d]+".tpl";
                            
                            if (scattered) {
                                sourcecode = "";
                            }
                            sourcecode += generate_source(
                                template_fn, symbol, bhopcode_to_cexpr(id),
                                type_out, type_in1, type_in2,
                                string(structure), license, include
                            );
                            if (scattered) {
                                target.compile(symbol, sourcecode.c_str(), sourcecode.size());
                            }
                        }
                        symbol_table.push_back(symbol);
                    }
                }
                if (!scattered) {
                    license = false;
                    include = false;
                }
            } else if ((!elementwise) && (!system_opcode)) {        // REDUCTION
                symbol = opcode;
                symbol += "_DD_" + string(signature);
                if (!target.symbol_ready(symbol)) {
                    template_fn = string(template_path) + "/reduction.tpl";
                    if (scattered) {
                        sourcecode = "";
                    }
                    sourcecode += generate_source(
                        template_fn, symbol, bhopcode_to_cexpr(id),
                        type_out, type_in1, type_in2,
                        string(structure), license, include
                    );
                    if (scattered) {
                        target.compile(symbol, sourcecode.c_str(), sourcecode.size());
                    }
                }
                symbol_table.push_back(symbol);
                if (!scattered) {
                    license = false;
                    include = false;
                }
            }
        }
    }

    if (!scattered) {
        std::string library_fn("symbols");
        std::ofstream symbols(target.lib_path(library_fn.c_str(), "ind"));
        for(
            std::vector<std::string>::iterator it = symbol_table.begin();
            it != symbol_table.end();
            ++it)
        {
            symbols << *it << std::endl;
        }
        symbols.close();

        target.compile(library_fn, sourcecode.c_str(), sourcecode.size());
        target.src_to_file(library_fn, sourcecode.c_str(), sourcecode.size()); 
    }

}

int main(int argc, char **argv)
{
    /*
    precompile(
        "gcc -shared -Wall -O2 -march=native -fopenmp -fPIC -std=c99 -x c - -lm -o ",
        "/home/safl/Desktop/bohrium/core/codegen/opcodes.json",
        "./objects",
        "./kernels",
        "./templates",
        true
    );*/

    precompile(
        "gcc -shared -Wall -O2 -march=native -fopenmp -fPIC -std=c99 -x c - -lm -o ",
        "/home/safl/Desktop/bohrium/core/codegen/opcodes.json",
        "./objects",
        "./kernels",
        "./templates",
        false
    );

    return 0;
}

