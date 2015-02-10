#include <stdexcept>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <map>

using namespace std;

namespace Plaid {

enum states {
    NONE,
    CURLY_OPEN,
    CURLY_CLOSE,
    PLACE_OPEN,
    PLACE_CLOSE,
    PLACE
};

class Plaid
{
public:
    Plaid(void) {}

    /**
    *   Add template from string.
    *   @param name
    *   @param tmpl
    */
    void add_from_string(string name, string tmpl)
    {
        templates_[name] = tmpl;
    }

    /**
    *   Add template from file.
    *
    *   @param filepath Full path of 
    *
    */
    void add_from_file(string name, string filepath)
    {
        ifstream ifs(filepath.c_str());
        
        string content(
            (istreambuf_iterator<char>(ifs)),
            (istreambuf_iterator<char>())
        );
        templates_[name] = content;
    }
     
    string fill(string name, map<string, string>& subjects)
    {
        string tmpl = templates_[name];

        stringstream place;
        int place_open, place_close;

        states state = NONE;
        for(int idx=0; idx<tmpl.size(); ++idx) {
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
                case PLACE_OPEN:
                    place_open = idx-1;
                    place_close = 0;
                    place.str("");
                    place.clear();
                    break;
                case PLACE:
                    place << token;
                    break;
                case PLACE_CLOSE:
                    place_close = idx;
                    tmpl.replace(place_open, (place_close-place_open)+1, subjects[place.str()]);
                    idx = place_open-1;
                    break;
            }
        }
        return tmpl;
    }

private:
    map<string, string> templates_;

};

}
