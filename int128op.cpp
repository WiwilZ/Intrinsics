#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include <cstddef>
#include <cstdint>

#include <bit>

template <typename T>
struct DivResult {
    T quotient;
    T remainder;
};

struct int128 {
    int64_t low;
    int64_t high;
};

struct uint128 {
    uint64_t low;
    uint64_t high;
};


namespace Naive {
    constexpr uint64_t Add(uint64_t x, bool& carry) noexcept {
        const auto result = x + carry;
        carry = result == 0;
        return result;
    }
    constexpr uint64_t Add(uint64_t x, uint64_t y, bool& carry) noexcept {
        const auto result = x + y + carry;
        carry = result <= x || result <= y;
        return result;
    }

    constexpr uint64_t Sub(uint64_t x, bool& borrow) noexcept {
        const auto result = x - borrow;
        borrow = result == -1;
        return result;
    }

    constexpr uint64_t Sub(uint64_t x, uint64_t y, bool& borrow) noexcept {
        const auto result = x - y - borrow;
        borrow = result >= x;
        return result;
    }

    constexpr uint64_t MulHigh(uint64_t x, uint64_t y) noexcept {
        /*
        |  d |  c |  b |  a |
        128  96   64   32   0
             |    | xl * yl |
             | xh * yl |
             | xl * yh |
        | xh * yh |
        */
        const uint64_t xh = x >> 32;
        const uint64_t xl = x & 0xFFFFFFFF;
        const uint64_t yh = y >> 32;
        const uint64_t yl = y & 0xFFFFFFFF;

        const uint64_t b0 = (xl * yl) >> 32;
        const uint64_t bc0 = b0 + xh * yl;
        const uint64_t b1 = bc0 & 0xFFFFFFFF;
        const uint64_t c0 = bc0 >> 32;
        const uint64_t bc1 = b1 + xl * yh;
        const uint64_t c1 = c0 + (bc1 >> 32);
        return c1 + xh * yh;
    }

    constexpr uint128 Mul(uint64_t x, uint64_t y) noexcept {
        const uint64_t xh = x >> 32;
        const uint64_t xl = x & 0xFFFFFFFF;
        const uint64_t yh = y >> 32;
        const uint64_t yl = y & 0xFFFFFFFF;

        const uint64_t xlyl = xl * yl;
        const uint64_t a0 = xlyl & 0xFFFFFFFF;
        const uint64_t b0 = xlyl >> 32;
        const uint64_t bc0 = b0 + xh * yl;
        const uint64_t b1 = bc0 & 0xFFFFFFFF;
        const uint64_t c0 = bc0 >> 32;
        const uint64_t bc1 = b1 + xl * yh;
        const uint64_t c1 = c0 + (bc1 >> 32);
        return {(bc1 << 32) | a0, c1 + xh * yh};
    }

    constexpr DivResult<uint64_t> DivHelper(uint64_t xbc, uint32_t xa, uint64_t y) noexcept {
        /*
        xc  xb  xa
            yb  ya

        q = [xbc xa] / y
        estimate q = xbc / yb ... r  =>  xbc = q * yb + r
        guarantee 
           q * y
        =  (q * yb << 32) + q * ya 
        <= (xbc << 32) + xa 
        =  (q * yb << 32) + (r << 32) + xa
        */
        const uint32_t yb = y >> 32;
        const uint32_t ya = y & 0xFFFFFFFF;

        uint64_t q = xbc / yb;
        uint64_t r = xbc % yb;
        uint64_t t0 = q * ya;
        uint64_t t1 = (r << 32) | xa;
        if (t0 > t1) {
            q -= ((t0 - t1) > y) + 1;
        }
        r = ((xbc << 32) | xa) - q * y; // remainder is guaranteed to be 64 bits
        
        return {q, r}
    }

    constexpr DivResult<uint64_t> Div(uint64_t xh, uint64_t xl, uint64_t y) noexcept {
        assert(xh < y && y != 0);

        if (xh == 0) {
            return {xl / y, xl % y};
        }

        const auto shift = std::countl_zero(y);
        if (shift > 0) {
            y <<= shift;
            xh = (xh << shift) | (xl >> (64 - shift));
            xl <<= shift;
        }

        auto&& [qh, r0] = DivHelper(xh, xl >> 32, y);
        auto&& [ql, r] = DivHelper(r0, xl & 0xFFFFFFFF, y);
        return {(qh << 32) | ql, r >> shift};
    }
}



constexpr uint64_t Add(uint64_t a, uint64_t b, bool& carry) noexcept {
#ifdef __SIZEOF_INT128__
    const auto result = static_cast<__uint128_t>(a) + b + carry;
    carry = result >> 64;
    return result;
#elif defined(_MSC_VER) 
    uint64_t result;
    carry = _addcarryx_u64(carry, a, b, &result);
    return result;
#else
    const auto result = a + b + carry;
    carry = result <= a || result <= b;
    return result;
#endif
}


#if defined(_MSC_VER) && !defined(__clang__)
/*寄存器传参顺序：rcx rdx r8 r9*/

int64_t MulHigh(int64_t a, int64_t b) { 
    return __mulh(a, b);
}

uint64_t MulHigh(uint64_t a, uint64_t b) {
    // return __umulh(a, b);
    uint64_t high;
    _mulx_u64(a, b, &high);
    return high;
}

int128 Mul(int64_t a, int64_t b) {
    int64_t high;
    const int64_t low = _mul128(a, b, &high);
    return {low, high};
}

uint128 Mul(uint64_t a, uint64_t b) {
    uint64_t high;
    // const uint64_t low = _umul128(a, b, &high);
    const uint64_t low = _mulx_u64(a, b, &high);
    return {low, high};
}

DivResult<int64_t> Div(int64_t ah, int64_t al, int64_t b) {
    int64_t remainder;
    const int64_t quotient = _div128(ah, al, b, &remainder);
    return {quotient, remainder};
}

DivResult<uint64_t> Div(uint64_t ah, uint64_t al, uint64_t b) {
    uint64_t remainder;
    const uint64_t quotient = _udiv128(ah, al, b, &remainder);
    return {quotient, remainder};
}

#else
/*寄存器传参顺序：rdi rsi rdx rcx r8 r9*/

int64_t MulHigh(int64_t a, int64_t b) {
    return (static_cast<__int128_t>(a) * b) >> 64;
}

uint64_t MulHigh(uint64_t a, uint64_t b) {
    return (static_cast<__uint128_t>(a) * b) >> 64;
}

__int128_t Mul(int64_t a, int64_t b) { 
    /*
    // imul 乘数
    // 另一乘数为rax
    // 结果为rdx:rax
    int128 result;
    __asm__("imulq %[a]"
            : "=d"(result.high), "=a"(result.low)
            : [a] "r"(a), "a"(b)
            : "cc");
    return result;
    */
    return static_cast<__int128_t>(a) * b; 
}

__uint128_t Mul(uint64_t a, uint64_t b) {
    /*
    // mulx 积的高位(dst1), 积的低位(dst2), 乘数(src)
    // 另一乘数为rdx
    uint128 result;
    __asm__("mulx %[a], %%rax, %%rdx"
            : "=d"(result.high), "=a"(result.low)
            : [a] "r"(a), "d"(b)
            : "cc");
    return result;
    */
    return static_cast<__uint128_t>(a) * b;
}

DivResult<int64_t> Div(int64_t ah, int64_t al, int64_t b) {
    // idiv 除数
    // 被除数为rdx:rax
    // 余数为rdx，商为rax
    DivResult<int64_t> result;
    __asm__("idivq %[b]"
            : "=d"(result.remainder), "=a"(result.quotient)
            : "d"(ah), "a"(al), [b] "r"(b)
            : "cc");
    return result;
}

DivResult<uint64_t> Div(uint64_t ah, uint64_t al, uint64_t b) {
    DivResult<uint64_t> result;
    __asm__("divq %[b]"
            : "=d"(result.remainder), "=a"(result.quotient)
            : "d"(ah), "a"(al), [b] "r"(b)
            : "cc");
    return result;
}
#endif


#include <iostream>


int main() {
    auto res = Div(1ul, 5ul, 2ul);
    std::cout << res.quotient << " " << res.remainder << "\n";
}