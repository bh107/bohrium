#include <iostream>

template<typename LeftIter, typename RightIter>
::testing::AssertionResult CheckEqualCollections(LeftIter left_begin,
                                                 LeftIter left_end,
                                                 RightIter right_begin)
{
    std::stringstream message;
    std::size_t index(0);
    bool equal(true);

    for(;left_begin != left_end; left_begin++, right_begin++) {
        if (*left_begin != *right_begin) {
            equal = false;
            message << "\n  Mismatch in position " << index << ": ";
            message << *left_begin << " != " <<  *right_begin;
        }
        ++index;
    }
    if (message.str().size()) {
        message << "\n";
    }
    return equal ? ::testing::AssertionSuccess() :
                   ::testing::AssertionFailure() << message.str();
}

