#ifndef TIMER_H
#define TIMER_H

#include <chrono>
#include <cstdio>

// RAII 计时器：构造时记录开始时间，析构时计算并打印耗时。
// 为什么用 RAII？利用作用域自动析构，不会忘记计停止，异常安全。
// 为什么用 chrono 而不是 clock()？chrono 是高精度、单调时钟，不受系统时间调整影响。
class Timer {
public:
    // explicit 防止隐式转换（避免 const char* 自动构造 Timer）。
    explicit Timer(const char* label) : label_(label) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        // duration_cast 转成微秒（microseconds = 10^-6 秒）。
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        printf("[%s] %.3f ms\n", label_, us / 1000.0);
    }

    // 禁用拷贝。计时器是"一次性"对象，拷贝语义没意义且容易出错。
    Timer(const Timer&) = delete;
    Timer& operator=(const Timer&) = delete;

private:
    const char* label_;
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// 更精确的计时函数：返回毫秒，供 benchmark 多次取最小值用。
// 为什么单独提供？因为有时你想手动控制多次循环取最小值，RAII 版本不方便。
inline double elapsed_ms(std::chrono::time_point<std::chrono::high_resolution_clock> start,
                         std::chrono::time_point<std::chrono::high_resolution_clock> end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
}

// 便捷获取当前时间点
inline std::chrono::time_point<std::chrono::high_resolution_clock> now() {
    return std::chrono::high_resolution_clock::now();
}

#endif // TIMER_H
