#include <onnxruntime_cxx_api.h>
#include <raylib.h>
#include "AudioFile.h"
#include "slice.hpp"
#include "oouras_real_fft.hpp"
#include "resample_iir.hpp"
#include "resample_coeffs.h"

Color GetSpectrumColor(float normal) {
    normal = fmaxf(0.0f, fminf(1.0f, normal));
    
    if (normal < 0.3f) {
        float t = (normal - 0.0f) / (0.3f - 0.0f);
        return ColorLerp(BLACK, BLUE, t);
    }
    else if (normal < 0.6f) {
        float t = (normal - 0.3f) / (0.6f - 0.3f);
        return ColorLerp(BLUE, MAGENTA, t);
    }
    else if (normal < 0.8f) {
        float t = (normal - 0.6f) / (0.8f - 0.6f);
        return ColorLerp(MAGENTA, YELLOW, t);
    }
    else {
        float t = (normal - 0.8f) / (1.0f - 0.8f);
        return ColorLerp(YELLOW, WHITE, t);
    }
}

constexpr float kSpectrumFloorDb = -100.0f;
constexpr float kSpectrumTopDb = 10.0f;
constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr float kConfidence = 0.9f;

constexpr auto kAudioPath = "../../working/o.wav";
constexpr auto kModelPath = L"../../model.onnx";

int main() {
    // loading files
    AudioFile<float> infile;
    bool audio_file_loaded = infile.load(kAudioPath);
    if (!audio_file_loaded) {
        return 1;
    }

    // resample
    std::vector<float> input_data;
    if (infile.getSampleRate() != 16000) {
        qwqdsp::fx::ResampleIIR<qwqdsp::fx::coeff::MedianCoeffs<float>, 127> resampler;
        resampler.Init(infile.getSampleRate(), 16000);
        input_data = resampler.Process<float>(infile.samples.front());
    }
    else {
        input_data = infile.samples.front();
    }

    // Swift_F0 detect pitch
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SwiftF0");
    Ort::SessionOptions session_options;
 
    Ort::Session session(env, kModelPath, session_options);
 
    auto input_name = session.GetInputNames().front();
    auto output_name = session.GetOutputNames();
 
    int64_t batch_size = 1;  
    std::vector<int64_t> input_shape = {batch_size, static_cast<int64_t>(input_data.size())}; 

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name[0].c_str(), output_name[1].c_str()};

    auto output_tensors = session.Run(
        Ort::RunOptions{},
        input_names, &input_tensor, 1,
        output_names, 2
    );

    // get output
    auto* pitch_ptr = output_tensors[0].GetTensorMutableData<float>();
    auto shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto shape = shape_info.GetShape();
    auto num_frames = shape[1];
    auto* confidence_ptr = output_tensors[1].GetTensorMutableData<float>();

    // draw audio as spectrum and pitch
    InitWindow(kWindowWidth, kWindowHeight, "swift_f0_cpp");
    qwqdsp::segement::Slice1D<float> slice{input_data};
    qwqdsp::spectral::OourasRealFFT fft;

    // this matchs model's stft parameter
    constexpr size_t kFFTSize = 1024;
    constexpr size_t kNumBins = kFFTSize / 2 + 1;
    constexpr size_t kHopSize = 256;

    fft.Init(kFFTSize);
    auto img = GenImageColor(num_frames, kNumBins, BLACK);
    float min_db_gain = std::pow(10.0f, kSpectrumFloorDb / 20.0f);
    float fs = static_cast<float>(infile.getSampleRate());
    for (size_t i = 0; i < num_frames; ++i) {
        float x[kFFTSize];
        slice.Read(kFFTSize, kHopSize, x);
        float reim[kFFTSize+2];
        fft.FFT(x, reim);
        
        // draw spectrum
        float scale = 2.0f / fft.GetFFTSize();
        for (size_t j = 0; j < kNumBins; ++j) {
            float re = reim[2 * j];
            float im = reim[2 * j + 1];
            float g = std::sqrt(re * re + im * im) * scale;
            float db = kSpectrumFloorDb;
            if (g > min_db_gain) {
                db = 20.0f * std::log10(g);
            }
            float gain_normal = (db - kSpectrumFloorDb) / (kSpectrumTopDb - kSpectrumFloorDb);
            auto color = GetSpectrumColor(gain_normal);
            ImageDrawPixel(&img, i, kNumBins - 1 - j, color);
        }

        // draw pitch
        if (confidence_ptr[i] > kConfidence) {
            float pitch = pitch_ptr[i];
            float freq_normal = pitch / (fs / 2.0f);
            float y = (1.0f - freq_normal) * img.height;
            ImageDrawPixel(&img, i, y, WHITE);
            ImageDrawPixel(&img, i, y-1, BLACK);
            ImageDrawPixel(&img, i, y+1, BLACK);
        }
    }

    auto texture = LoadTextureFromImage(img);
    SetTargetFPS(30);
    while (!WindowShouldClose()) {
        BeginDrawing();
        DrawTexturePro(texture, Rectangle{0,0,static_cast<float>(texture.width),static_cast<float>(texture.height)}, Rectangle{0,0,static_cast<float>(GetScreenWidth()),static_cast<float>(GetScreenHeight())}, {0,0}, 0, WHITE);
        EndDrawing();
    }

    UnloadImage(img);
    UnloadTexture(texture);
    CloseWindow();
}