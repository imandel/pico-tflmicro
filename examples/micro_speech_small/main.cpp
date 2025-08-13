#include <stdio.h>
#include <string.h>
#include "pico/stdlib.h"

// TensorFlow Lite Micro includes
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "model_config.h"

inline float DequantizeInt8ToFloat(int8_t value, float scale, int zero_point)
{
    return static_cast<float>(value - zero_point) * scale;
}

void FillInputPattern(int8_t *buffer, size_t size)
{
    const int8_t pattern[] = {42, -42, 85, -85}; // 01010101, 10101010 patterns
    for (size_t i = 0; i < size; i++)
    {
        buffer[i] = pattern[i % 4];
    }
}


void Error_Handler(void)
{

    while (1) {
        sleep_ms(100);
    }
        
}

int main(void)
{
    stdio_init_all();
    
    // Brief delay to allow stdio to stabilize
    sleep_ms(1000);
    
    printf("\r\n=== Starting %s Model Test ===\r\n", ModelConfig::GetModelName());

    uint64_t program_start_time = time_us_64();

    // Get model from configuration
    const tflite::Model *model = tflite::GetModel(ModelConfig::GetModelData());
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        printf("The model version of %d does not match the version of the schema of version %d\r\n",
                    model->version(), TFLITE_SCHEMA_VERSION);
        Error_Handler();
    }

    sleep_ms(2000);
    printf("Program start time: %llu microseconds\r\n", program_start_time);

    // Create op resolver with model-specific operations
    tflite::MicroMutableOpResolver<ModelConfig::kOpResolverSize> resolver;
    ModelConfig::InitializeOpResolver(resolver);

    // Create interpreter
    tflite::MicroInterpreter interpreter(
        model, resolver, tensor_arena, ModelConfig::kTensorArenaSize);

    // Allocate tensors
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk)
    {
        printf("AllocateTensors() failed\r\n");
        Error_Handler();
    }

    // Get input and output tensors
    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    // Print model info
    printf("Model loaded successfully: %s\r\n", ModelConfig::GetModelName());
    printf("Input size: %d bytes\r\n", ModelConfig::GetInputSize());
    printf("Expected categories: %d\r\n", ModelConfig::GetCategoryCount());

    if (input->bytes != ModelConfig::GetInputSize())
    {
        printf("Input tensor size mismatch! Expected: %d, got: %d\r\n",
                    ModelConfig::GetInputSize(), input->bytes);
    }

    // Display tensor information
    printf("Input tensor type: %d\r\n", input->type);
    printf("Input tensor scale: %f\r\n", input->params.scale);
    printf("Input tensor zero point: %d\r\n", input->params.zero_point);

    printf("Output tensor type: %d\r\n", output->type);
    printf("Output tensor scale: %f\r\n", output->params.scale);
    printf("Output tensor zero point: %d\r\n", output->params.zero_point);

    // Report arena memory usage
    printf("Tensor arena memory used: %lu bytes\r\n", (unsigned long)interpreter.arena_used_bytes());
    printf("Tensor arena memory available: %d bytes\r\n", ModelConfig::kTensorArenaSize);

    printf("Initialization complete\r\n");
    uint64_t current_time = time_us_64();
    printf("Setup() end time: %llu\r\n", current_time);
    uint64_t setup_time = current_time - program_start_time;
    printf("Setup takes: %llu microseconds\r\n", setup_time);

    const int num_inferences = 10;
    uint64_t total_memcpy_time = 0;
    uint64_t total_inference_time = 0;
    uint64_t total_postprocess_time = 0;

    for (int i = 0; i < num_inferences; ++i)
    {
        uint64_t loop_start_time = time_us_64();
        printf("\r\nInference %d/%d starts at: %llu\r\n", i + 1, num_inferences, loop_start_time);

        uint64_t memcpy_start_time = time_us_64();
        // using all zeros for input data
        memset(input->data.int8, 0, input->bytes);
        uint64_t memcpy_end_time = time_us_64();
        uint64_t memcpy_time = memcpy_end_time - memcpy_start_time;
        total_memcpy_time += memcpy_time;
        printf("memcpy took: %llu microseconds\r\n", memcpy_time);

        uint64_t inference_start_time = time_us_64();
        TfLiteStatus invoke_status = interpreter.Invoke();
        if (invoke_status != kTfLiteOk)
        {
            printf("Invoke failed\r\n");
            sleep_ms(1000);
            continue;
        }
        uint64_t inference_end_time = time_us_64();
        uint64_t inference_time = inference_end_time - inference_start_time;
        total_inference_time += inference_time;
        printf("Inference took: %llu microseconds\r\n", inference_time);

        uint64_t postprocess_start_time = time_us_64();

        printf("Results: [");
        for (int j = 0; j < output->dims->data[1]; j++)
        {
            float converted = DequantizeInt8ToFloat(
                output->data.int8[j],
                output->params.scale,
                output->params.zero_point);

            printf("%.3f", converted);
            if (j < output->dims->data[1] - 1)
            {
                printf(", ");
            }
        }
        printf("]\r\n");

        int max_idx = 0;
        float max_val = DequantizeInt8ToFloat(
            output->data.int8[0],
            output->params.scale,
            output->params.zero_point);

        for (int j = 1; j < output->dims->data[1]; j++)
        {
            float val = DequantizeInt8ToFloat(
                output->data.int8[j],
                output->params.scale,
                output->params.zero_point);

            if (val > max_val)
            {
                max_val = val;
                max_idx = j;
            }
        }

        // Use the category labels from the model configuration
        const char **labels = ModelConfig::GetCategoryLabels();
        printf("Detected: %s (%.3f)\r\n", labels[max_idx], max_val);

        uint64_t postprocess_end_time = time_us_64();
        uint64_t postprocess_time = postprocess_end_time - postprocess_start_time;
        total_postprocess_time += postprocess_time;
        printf("Post-processing took: %llu microseconds\r\n", postprocess_time);
    }

    printf("\r\n--- Average latencies after %d inferences ---\r\n", num_inferences);
    printf("Model: %s\r\n", ModelConfig::GetModelName());
    printf("Average memcpy time: %.2f microseconds\r\n", (float)total_memcpy_time / num_inferences);
    printf("Average inference time: %.2f microseconds\r\n", (float)total_inference_time / num_inferences);
    printf("Average post-processing time: %.2f microseconds\r\n", (float)total_postprocess_time / num_inferences);
    uint64_t total_avg_time = (total_memcpy_time + total_inference_time + total_postprocess_time) / num_inferences;
    printf("Average total loop time: %llu microseconds\r\n", total_avg_time);

    while (1)
    {
        // Infinite loop - keep LED on to show program is running
        sleep_ms(1000);
    }

    return 0;
}