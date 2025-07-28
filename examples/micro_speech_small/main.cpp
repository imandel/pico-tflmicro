#include <stdio.h>
#include <limits>

#include "pico/stdlib.h"
#include "hardware/timer.h"

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Model data
#include "kws_model_data.h"
#include "kws_model_settings.h"

#define PROFILE_MICRO_SPEECH

// Globals
// Allocate tensor arena memory
constexpr int kTensorArenaSize = 28584;
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

// Function to dequantize int8 to float - matches benchmark implementation
inline float DequantizeInt8ToFloat(int8_t value, float scale, int zero_point)
{
    return static_cast<float>(value - zero_point) * scale;
}

// Function to fill input with a simple pattern - no computation overhead
void FillInputPattern(int8_t* buffer, size_t size) {

    const int8_t pattern[] = {42, -42, 85, -85};  // 01010101, 10101010 patterns
    for (size_t i = 0; i < size; i++) {
        buffer[i] = pattern[i % 4];
    }

}

int main()
{
    // Initialize the Pico
    stdio_init_all();

    // Wait for serial connection
    sleep_ms(2000);

    printf("Benchmarking Example with Pattern Data\n");
    unsigned long program_start_time = time_us_64();

    // Map the model into a usable data structure
    const tflite::Model *model = tflite::GetModel(g_kws_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        MicroPrintf("Model version mismatch!");
        return 1;
    }
    MicroPrintf("Program start time: %llu microseconds", program_start_time);

    tflite::MicroMutableOpResolver<4> resolver;
    resolver.AddReshape();
    resolver.AddFullyConnected();
    resolver.AddDepthwiseConv2D();
    resolver.AddSoftmax();

    // Create interpreter
    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);

    // Allocate tensors
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        MicroPrintf("AllocateTensors() failed");
        return 1;
    }

    // Get input and output tensors
    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    // Print model info
    printf("Model loaded successfully\n");
    printf("Input tensor shape: %d x %d x %d\n",
           kNumRows, kNumCols, kNumChannels);
    printf("Input size: %d bytes\n", kKwsInputSize);
    printf("Expected categories: %d\n", kCategoryCount);

    // Verify input dimensions match what we expect
    if (input->bytes != kKwsInputSize)
    {
        MicroPrintf("Input tensor size mismatch! Expected: %d, got: %d",
                    kKwsInputSize, input->bytes);
    }

    // Display tensor information
    printf("Input tensor type: %d\n", input->type);
    printf("Input tensor scale: %f\n", input->params.scale);
    printf("Input tensor zero point: %d\n", input->params.zero_point);

    printf("Output tensor type: %d\n", output->type);
    printf("Output tensor scale: %f\n", output->params.scale);
    printf("Output tensor zero point: %d\n", output->params.zero_point);

    // Report arena memory usage
    printf("Tensor arena memory used: %zu bytes\n", interpreter.arena_used_bytes());
    printf("Tensor arena memory available: %d bytes\n", kTensorArenaSize);

    MicroPrintf("Initialization complete");
    uint64_t current_time = time_us_64();
    MicroPrintf("Setup() end time: %llu", current_time);
    uint64_t setup_time = current_time - program_start_time;
    MicroPrintf("Setup takes: %llu microseconds", setup_time);

    // Benchmarking variables
#ifdef PROFILE_MICRO_SPEECH
    static uint32_t prof_count = 0;
    static uint64_t prof_sum = 0;
    static uint64_t prof_min = std::numeric_limits<uint64_t>::max();
    static uint64_t prof_max = 0;
#endif

    // Main inference loop
    while (true)
    {
        uint64_t loop_start_time = time_us_64();
        MicroPrintf("New loop starts at: %llu", loop_start_time);

        // Fill input with pattern data (minimal overhead)
        uint64_t data_fill_start_time = time_us_64();
        FillInputPattern(input->data.int8, input->bytes);
        uint64_t data_fill_end_time = time_us_64();
        uint64_t data_fill_time = data_fill_end_time - data_fill_start_time;
        MicroPrintf("Pattern data fill took: %llu microseconds", data_fill_time);

        // Run inference
        uint64_t inference_start_time = time_us_64();
        TfLiteStatus invoke_status = interpreter.Invoke();
        if (invoke_status != kTfLiteOk)
        {
            MicroPrintf("Invoke failed");
            sleep_ms(1000);
            continue;
        }
        uint64_t inference_end_time = time_us_64();
        uint64_t inference_time = inference_end_time - inference_start_time;

        MicroPrintf("Inference took: %llu microseconds", inference_time);

#ifdef PROFILE_MICRO_SPEECH
        // Update profiling statistics
        prof_count++;
        prof_sum += inference_time;
        if (inference_time < prof_min) prof_min = inference_time;
        if (inference_time > prof_max) prof_max = inference_time;

        if (prof_count % 100 == 0) {
            uint64_t prof_avg = prof_sum / prof_count;
            MicroPrintf("Inference stats after %u runs:", prof_count);
            MicroPrintf("  Average: %llu us", prof_avg);
            MicroPrintf("  Min: %llu us", prof_min);
            MicroPrintf("  Max: %llu us", prof_max);
        }
#endif

        // Start timing post-processing
        uint64_t postprocess_start_time = time_us_64();

        // Process results
        printf("Results: [");
        for (int i = 0; i < output->dims->data[1]; i++)
        {
            float converted = DequantizeInt8ToFloat(
                output->data.int8[i],
                output->params.scale,
                output->params.zero_point);

            printf("%.3f", converted);
            if (i < output->dims->data[1] - 1)
            {
                printf(", ");
            }
        }
        printf("]\n");

        // Find the keyword with highest probability
        int max_idx = 0;
        float max_val = DequantizeInt8ToFloat(
            output->data.int8[0],
            output->params.scale,
            output->params.zero_point);

        for (int i = 1; i < output->dims->data[1]; i++)
        {
            float val = DequantizeInt8ToFloat(
                output->data.int8[i],
                output->params.scale,
                output->params.zero_point);

            if (val > max_val)
            {
                max_val = val;
                max_idx = i;
            }
        }

        // Use the category labels from the settings
        printf("Detected keyword: %s (%.3f)\n", kCategoryLabels[max_idx], max_val);

        // End timing and report post-processing time
        uint64_t postprocess_end_time = time_us_64();
        MicroPrintf("Post-processing took: %llu microseconds", 
                   postprocess_end_time - postprocess_start_time);

        // Optional: Add small delay for readability
        // sleep_ms(100);
    }

    return 0;
}