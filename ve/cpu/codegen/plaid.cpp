#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>
#include "plaid.hpp"

using namespace std;

namespace kp{
namespace engine{
namespace codegen{

enum states {
    NONE,
    CURLY_OPEN,
    CURLY_CLOSE,
    PLACE_OPEN,
    PLACE_CLOSE,
    PLACE
};

Plaid::Plaid(string template_directory) : template_directory_(template_directory) {
    add_from_file("license",    "license.tpl");
    add_from_file("kernel",     "kernel.tpl");
    add_from_file("code.block", "code.block.tpl");
    add_from_file("code.loop",  "code.loop.tpl");
    add_from_file("scan.1d",    "scan.1d.tpl");
    add_from_file("scan.nd",    "scan.nd.tpl");
    add_from_file("walker.axis.nd",     "walker.axis.nd.tpl");
    #if defined(CAPE_WITH_OPENACC)
    add_from_file("walker.collapsed",   "walker.collapsed.acc.tpl");
	#else
	add_from_file("walker.collapsed",   "walker.collapsed.tpl");
	#endif
    add_from_file("walker.inner.2d",    "walker.inner.2d.tpl");
    add_from_file("walker.inner.nd",    "walker.inner.nd.tpl");
    add_from_file("walker.scalar",      "walker.scalar.tpl");
}

string Plaid::text(void)
{
    stringstream ss;
    ss << "Plaid {" << endl;
    ss << "  template_path = '" << template_directory_ << "'" << endl;
    ss << "}";
    return ss.str();
}

void Plaid::add_from_string(string name, string tmpl)
{
    templates_[name] = tmpl;
}

void Plaid::add_from_file(string name, string filename)
{
    stringstream ss;
    ss << template_directory_ << "/" << filename;
    string abspath = ss.str();
    std::ifstream ifs(abspath.c_str());

    string content(
        (istreambuf_iterator<char>(ifs)),
        (istreambuf_iterator<char>())
    );
    templates_[name] = content;
}

string Plaid::fill(string name, map<string, string>& subjects)
{
    string tmpl = templates_[name];

    stringstream place;
    unsigned int place_open = 0;
    unsigned int place_close = 0;

    states state = NONE;
    for(unsigned int idx=0; idx<tmpl.size(); ++idx) {
        char token = tmpl[idx];
        switch (state) {    // State transition
            case CURLY_OPEN:
                if ('{' == token) {
                    state = PLACE_OPEN;
                } else {
                    state = NONE;
                }
                break;

            case PLACE_OPEN:
                if ('{' == token) {
                    cout << "ERROR! Use {{ to place place-holder." << endl;
                }

                if ('}' == token) {
                    cout << "ERROR! Empty placeholder." << endl;
                }
                state = PLACE;
                break;

            case CURLY_CLOSE:
                if ('}' != token) {
                    cout << "ERROR! Incorrectly closed placeholder." << endl;
                    state = NONE;
                } else {
                    state = PLACE_CLOSE;
                }
                break;

            case PLACE:
                if ('}' == token) {
                    state = CURLY_CLOSE;
                }
                break;

            case PLACE_CLOSE:
                state = NONE;
            default:
                if ('{' == token) {
                    state = CURLY_OPEN;
                }
                break;
        }

        switch(state) { // State handling
            case PLACE_OPEN:        //  Reset placeholder
                place_open = idx-1;
                place_close = 0;
                place.str("");
                place.clear();
                break;
            case PLACE:             // Add token to placeholder
                place << token;
                break;
            case PLACE_CLOSE:       // Replace placeholder with subject
                place_close = idx;
                replace(
                    tmpl,
                    place_open,
                    (place_close-place_open)+1,
                    subjects[place.str()]
                );
                idx = place_open-1; // Continue after the inserted string
                break;
            default:
                break;
        }
    }
    return tmpl;
}

unsigned int Plaid::indentlevel(string text, unsigned int index)
{
    for(int idx=index; idx>=0; --idx) {
        char token = text[idx];
        if ('\n' == token) {
            return (index-idx-1);
        }
    }
    return index;
}

void Plaid::replace(string& tmpl, unsigned int begin, unsigned int count, string subject)
{
    stringstream ss;    // Create indentation replacement string
    ss << '\n';
    for(unsigned int lvl=0; lvl<indentlevel(tmpl, begin); ++lvl) {
        ss << ' ';
    }
    string indentation = ss.str();

    for(unsigned int idx=0; idx<subject.size(); ++idx) { // Indent
        if ('\n' == subject[idx]) {
            subject.replace(idx, 1, indentation);
        }
    }

    tmpl.replace(begin, count, subject);
}

}}}
