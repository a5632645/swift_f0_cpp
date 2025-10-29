#pragma once
#include <complex>
#include <cstddef>
#include <numbers>
#include <numeric>
#include <vector>
#include "hamming.hpp"
#include "helper.hpp"
#include "oouras_real_fft.hpp"

namespace qwqdsp::spectral {
class ReassignmentCorrect {
public:
    void Init(size_t fft_size) {
        fft_.Init(fft_size);
        buffer_.resize(fft_size);
        xh_data_.resize(fft_size + 2);
        xdh_data_.resize(fft_size + 2);
        xth_data_.resize(fft_size + 2);
        window_.resize(fft_size);
        dwindow_.resize(fft_size);
        twindow_.resize(fft_size);
        ChangeWindow([](auto win, auto dwin) {
            window::Hamming::Window(win, true);
            window::Hamming::DWindow(dwin);
        });
    }

    /**
     * @tparam func void(std::span<float> window, std::span<float> dwindow)
     */
    template<class Func>
    void ChangeWindow(Func&& func) noexcept(noexcept(func(std::declval<std::span<float>>(), std::declval<std::span<float>>()))) {
        func(std::span<float>{window_}, std::span<float>{dwindow_});
        window::Helper::TWindow(twindow_, window_);
        window_scale_ = window::Helper::NormalizeGain(window_);
        dwindow_scale_ = window_scale_ / (2.0f * std::numbers::pi_v<float>);
    }

    void Process(std::span<const float> time) noexcept {
        const size_t fft_size = fft_.GetFFTSize();
        for (size_t i = 0; i < fft_size; ++i) {
            buffer_[i] = time[i] * window_[i];
        }
        fft_.FFT(buffer_.data(), xh_data_.data());

        for (size_t i = 0; i < fft_size; ++i) {
            buffer_[i] = time[i] * dwindow_[i];
        }
        fft_.FFT(buffer_.data(), xdh_data_.data());

        for (size_t i = 0; i < fft_size; ++i) {
            buffer_[i] = time[i] * twindow_[i];
        }
        fft_.FFT(buffer_.data(), xth_data_.data());
    }

    float GetFrequency(size_t idx) const noexcept {
        auto xdh = std::complex{xdh_data_[2*idx],xdh_data_[2*idx+1]} * dwindow_scale_;
        auto xh = std::complex{xh_data_[2*idx],xh_data_[2*idx+1]} * window_scale_;
        auto up = xdh.imag() * xh.real() - xdh.real() * xh.imag();
        auto down = std::norm(xh);
        auto freq_c = -up / down;
        return (idx + freq_c) / static_cast<float>(fft_.GetFFTSize());
    }

    void GetFrequency(std::span<float> freq) const noexcept {
        for (size_t i = 0; i < freq.size(); ++i) {
            freq[i] = GetFrequency(i);
        }
    }

    float GetGain(size_t idx) const noexcept {
        return std::abs(std::complex{xh_data_[2*idx], xh_data_[2*idx+1]}) * window_scale_;
    }

    void GetGain(std::span<float> gain) const noexcept {
        for (size_t i = 0; i < gain.size(); ++i) {
            gain[i] = GetGain(i);
        }
    }
private:
    OourasRealFFT fft_;
    std::vector<float> buffer_;
    std::vector<float> window_;
    std::vector<float> dwindow_;
    std::vector<float> twindow_;
    std::vector<float> xh_data_;
    std::vector<float> xdh_data_;
    std::vector<float> xth_data_;
    float window_scale_{};
    float dwindow_scale_{};
};
}