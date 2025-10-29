#pragma once

#include <iostream>
#include <string>
#include <algorithm>

template <typename T>
T math_mod(T a, T n) {
    if (n == 0) {
        throw std::invalid_argument("Modulo by zero");
    }
    if (n < 0) {
        n = -n;
    }
    T r = a % n;
    return r < 0 ? r + n : r;
}


template<typename T>
inline T floor_div(T a, T b) {
    if (b == 0) {
        throw std::invalid_argument("Division by zero");
    }
    T q = a / b;
    T r = a % b;
    if ((r != 0) && ((r < 0) != (b < 0))) {
        q -= 1;
    }
    return q;
}

std::ostream& operator<<(std::ostream& os, __int128_t value) {
    if (value == 0) {
        os << "0";
        return os;
    }
    std::string str;
    bool is_negative = false;
    if (value < 0) {
        is_negative = true;
    }
    while (value != 0) {
        int digit;
        if (is_negative) {
            digit = -(value % 10);
            value /= 10;
        } else {
            digit = value % 10;
            value /= 10;
        }
        str += (char)('0' + digit);
    }
    if (is_negative) {
        str += '-';
    }
    std::reverse(str.begin(), str.end());
    os << str;
    return os;
}

std::ostream& operator<<(std::ostream& os, unsigned __int128 n) {
    if (n == 0) {
        os << "0";
        return os;
    }
    std::string str;
    while (n != 0) {
        str += (char)('0' + (n % 10));
        n /= 10;
    }
    std::reverse(str.begin(), str.end());
    os << str;
    return os;
}
