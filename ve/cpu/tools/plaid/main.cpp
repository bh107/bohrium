#include "plaid.hpp"

int main(void)
{
    Plaid::Plaid plaid;
    map<string, string> subjects;
    subjects["NAME"] = "simon";
    subjects["COOL"] = "hot";
    plaid.add_from_string("hello", "hello {{NAME}}, how {{COOL}} are {{HMM}} you?");
    cout << plaid.fill("hello", subjects) << endl;
    return 0;
}
