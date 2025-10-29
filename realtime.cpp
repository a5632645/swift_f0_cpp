#include <raylib.h>
#include <array>
#include <vector>
#include <semaphore>
#include <onnxruntime_cxx_api.h>

#include "miniaudio.h"
#include "reassignment.hpp"

static std::array<float, 2048> audio_buffer{};
static size_t wpos{};
static std::binary_semaphore audio_lock{1};

void MyAudioCallback(ma_device *pDevice, void *pOutput, const void *pInput, ma_uint32 frameCount) {
    audio_lock.acquire();
    const float* src = reinterpret_cast<const float*>(pInput);
    while (frameCount--) {
        audio_buffer[wpos++] = *src++;
        wpos &= 2047;
    }
    audio_lock.release();
}

constexpr float kConfidence = 0.9f;
constexpr int kWindowWidth = 1280;
constexpr int kWindowHeight = 720;
constexpr int kImageWidth = 192;
constexpr int kImageHeight = 512;
constexpr size_t kFftSize = 1024;
constexpr float kSampleRate = 16000.0f;
constexpr size_t kNumBins = kFftSize / 2 + 1;
constexpr float kSpectrumFloorDb = -60.0f;
constexpr float kSpectrumTopDb = 10.0f;
static float min_db_gain = std::pow(10.0f, kSpectrumFloorDb / 20.0f);

static float FreqToY(float freq) {
    static float min_pitch = std::log(20.0f);
    static float max_pitch = std::log(8000.0f);

    if (freq < 20.0f) return 0;
    if (freq > 80000.0f) return kImageHeight;

    float log = std::log(freq);
    float normal = (log - min_pitch) / (max_pitch - min_pitch);
    return normal * kImageHeight;
}

static RenderTexture2D texture_spectrum;
static RenderTexture2D texture_spectrum2;
static qwqdsp::spectral::ReassignmentCorrect fft;

static float audio_segement[1024]{};

constexpr auto kModelPath = L"../../model.onnx";

static float ProcessPitch() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "SwiftF0");
    static Ort::SessionOptions session_options;
    static Ort::Session session(env, kModelPath, session_options);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    auto input_name = session.GetInputNames().front();
    auto output_name = session.GetOutputNames();
 
    int64_t batch_size = 1;  
    int64_t input_shape[]{batch_size, static_cast<int64_t>(kFftSize)}; 

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, audio_segement, kFftSize,
        input_shape, 2);

    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name[0].c_str(), output_name[1].c_str()};

    auto output_tensors = session.Run(
        Ort::RunOptions{},
        input_names, &input_tensor, 1,
        output_names, 2
    );

    auto* pitch_ptr = output_tensors[0].GetTensorMutableData<float>();
    auto* confidence_ptr = output_tensors[1].GetTensorMutableData<float>();
    auto max_confidence_it = std::max_element(confidence_ptr, confidence_ptr + 4);
    size_t idx = max_confidence_it - confidence_ptr;
    if (*max_confidence_it > kConfidence) {
        return pitch_ptr[idx];
    }
    else {
        return 0;
    }
}

static Color GetSpectrumColor(float normal) {
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

static void DrawSpectrumAndPitch() {
    {
        audio_lock.acquire();
        size_t rpos = wpos - 1 - kFftSize;
        for (size_t i = 0; i < kFftSize; ++i) {
            rpos &= 2047;
            audio_segement[i] = audio_buffer[rpos++];
        }
        audio_lock.release();
    }

    BeginTextureMode(texture_spectrum);
        ClearBackground(BLANK);
        float pitch = ProcessPitch();
        if (pitch == 0.0f) {
            auto x = texture_spectrum.texture.width - 1;
            DrawLine(x, 0, x, kImageHeight, Color{32,32,32,255});
        }
        DrawTexture(texture_spectrum2.texture, 0, 0, WHITE);

        // draw spectrum
        fft.Process(audio_segement);
        float gains[kNumBins]{};
        float freqs[kNumBins]{};
        fft.GetFrequency(freqs);
        fft.GetGain(gains);
        float filter[kImageHeight]{};
        for (size_t j = 0; j < kNumBins; ++j) {
            if (!std::isfinite(freqs[j])) continue;

            float db = kSpectrumFloorDb;
            if (gains[j] > min_db_gain) {
                db = 20.0f * std::log10(gains[j]);
            }
            float gain_normal = (db - kSpectrumFloorDb) / (kSpectrumTopDb - kSpectrumFloorDb);
            auto color = GetSpectrumColor(gain_normal);
            size_t idx = FreqToY(freqs[j] * kSampleRate);
            if (gains[j] > filter[idx]) {
                filter[idx] = gains[j];
                DrawPixel(texture_spectrum.texture.width - 1, idx, color);
            }
        }
        // draw pitch
        if (pitch != 0.0f) {
            size_t idx = FreqToY(pitch);
            DrawPixel(texture_spectrum.texture.width - 1, idx + 1, BLACK);
            DrawPixel(texture_spectrum.texture.width - 1, idx, WHITE);
            DrawPixel(texture_spectrum.texture.width - 1, idx - 1, BLACK);
        }
    EndTextureMode();

    BeginTextureMode(texture_spectrum2);
        ClearBackground(BLANK);
        DrawTextureRec(texture_spectrum.texture, {1, 0, (float)texture_spectrum.texture.width-1, (float)texture_spectrum.texture.height}, {0, 0}, WHITE);
    EndTextureMode();

    DrawTexturePro(texture_spectrum.texture, Rectangle{0,0,(float)texture_spectrum.texture.width, (float)texture_spectrum.texture.height}, Rectangle{0,0,(float)kWindowWidth,(float)kWindowHeight}, Vector2{0,0}, 0, WHITE);
}

static ma_context audio_context;
static ma_device audio_device;

int main(int argc, char const *argv[]) {
    if (ma_context_init(NULL, 0, NULL, &audio_context) != MA_SUCCESS) {
        return -1;
    }

    ma_device_info *pPlaybackInfos;
    ma_uint32 playbackCount;
    ma_device_info *pCaptureInfos;
    ma_uint32 captureCount;
    if (ma_context_get_devices(&audio_context, &pPlaybackInfos, &playbackCount, &pCaptureInfos, &captureCount) != MA_SUCCESS) {
        return 1;
    }

    ma_device_config config;
    config = ma_device_config_init(ma_device_type_capture);
    config.capture.format = ma_format_f32; // Set to ma_format_unknown to use the device's native format.
    config.capture.channels = 1;           // Set to 0 to use the device's native channel count.
    config.sampleRate = kSampleRate;             // Set to 0 to use the device's native sample rate.
    config.dataCallback = MyAudioCallback;   // This function will be called when miniaudio needs more data.
    config.pUserData = nullptr;  // Can be accessed from the device object (device.pUserData).
    config.capture.pDeviceID = &pCaptureInfos[1].id;

    if (ma_device_init(NULL, &config, &audio_device) != MA_SUCCESS) {
        return -1;
    }
    ma_device_start(&audio_device); // The device is sleeping by default so you'll need to start it manually.

    InitWindow(kWindowWidth, kWindowHeight, "Simple XY");
    SetTargetFPS(30);

    texture_spectrum = LoadRenderTexture(kImageWidth, kImageHeight);
    texture_spectrum2 = LoadRenderTexture(kImageWidth, kImageHeight);
    fft.Init(kFftSize);

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);
        DrawSpectrumAndPitch();
        EndDrawing();
    }

    UnloadRenderTexture(texture_spectrum);
    UnloadRenderTexture(texture_spectrum2);
    CloseWindow();
    ma_device_uninit(&audio_device);
}
