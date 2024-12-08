#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h> // AVX intrinsics

float dot(float *a, float *b, size_t size) {
    float sum = 0;

    for (size_t i = 0; i < size; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

float dot_avx_256(float *a, float *b, size_t size) {
    __m256 sum = _mm256_setzero_ps();

    // AVX 256-bit (32 bytes) registers can hold simd_width_size (8) single-precision floats
    size_t simd_width_size = 256 / 8 / sizeof(float);
    size_t simd_iterations = size / simd_width_size;
    size_t simd_remainder = size % simd_width_size;

    for (size_t i = 0; i < simd_iterations; ++i) {
        __m256 a_vec = _mm256_loadu_ps(&a[i * simd_width_size]);
        __m256 b_vec = _mm256_loadu_ps(&b[i * simd_width_size]);

        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }

    // Handle the remainder using masking
    if (simd_remainder > 0) {
        int mask_bits[8] = {}; // All bits are initially set to 0
        for (size_t i = 0; i < simd_remainder; ++i) {
            mask_bits[i] = 0xFFFFFFFF;
        }

        __m256i mask = _mm256_setr_epi32(mask_bits[0], mask_bits[1], mask_bits[2], mask_bits[3], mask_bits[4], mask_bits[5], mask_bits[6], mask_bits[7]);

        __m256 a_vec = _mm256_maskload_ps(&a[simd_iterations * simd_width_size], mask);
        __m256 b_vec = _mm256_maskload_ps(&b[simd_iterations * simd_width_size], mask);

        sum = _mm256_add_ps(sum, _mm256_mul_ps(a_vec, b_vec));
    }

    float result[8];
    _mm256_storeu_ps(result, sum);

    return result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7];
}

#ifdef __AVX512F__
float dot_avx_512(float *a, float *b, size_t size) {
    __m512 sum = _mm512_setzero_ps();

    // AVX 512-bit (64 bytes) registers can hold simd_width_size (16) single-precision floats
    size_t simd_width_size = 512 / 8 / sizeof(float);
    size_t simd_iterations = size / simd_width_size;
    size_t simd_remainder = size % simd_width_size;

    for (size_t i = 0; i < simd_iterations; ++i) {
        __m512 a_vec = _mm512_loadu_ps(&a[i * simd_width_size]);
        __m512 b_vec = _mm512_loadu_ps(&b[i * simd_width_size]);

        sum = _mm512_add_ps(sum, _mm512_mul_ps(a_vec, b_vec));
    }

    // Handle the remainder using masking
    if (simd_remainder > 0) {
        __mmask16 mask = (1 << simd_remainder) - 1;

        __m512 a_vec = _mm512_maskz_loadu_ps(mask, &a[simd_iterations * simd_width_size]);
        __m512 b_vec = _mm512_maskz_loadu_ps(mask, &b[simd_iterations * simd_width_size]);

        sum = _mm512_add_ps(sum, _mm512_mul_ps(a_vec, b_vec));
    }

    float result[16];
    _mm512_storeu_ps(result, sum);

    return result[0] + result[1] + result[2] + result[3] + result[4] + result[5] + result[6] + result[7] + result[8] + result[9] + result[10] + result[11] + result[12] + result[13] + result[14] + result[15];
}
#endif

#ifdef TEST_MODE
void test_dot() {
    size_t size = 21;
    float a[size], b[size];

    for (size_t i = 0; i < size; ++i) {
        a[i] = (float)(i + 1);
        b[i] = (float)(i + 22);
    }

    float expected = 8162.0;
    float actual = dot(a, b, size);

    printf(expected == actual ? "\tPASSED\n" : "\tFAILED\n");
    printf("\tExpected: %f, Actual: %f\n", expected, actual);
}

void test_dot_avx_256() {
    size_t size = 21;
    float a[size], b[size];

    for (size_t i = 0; i < size; ++i) {
        a[i] = (float)(i + 1);
        b[i] = (float)(i + 22);
    }

    float expected = 8162.0;
    float actual = dot_avx_256(a, b, size);

    printf(expected == actual ? "\tPASSED\n" : "\tFAILED\n");
    printf("\tExpected: %f, Actual: %f\n", expected, actual);
}

#ifdef __AVX512F__
void test_dot_avx_512() {
    size_t size = 21;
    float a[size], b[size];

    for (size_t i = 0; i < size; ++i) {
        a[i] = (float)(i + 1);
        b[i] = (float)(i + 22);
    }

    float expected = 8162.0;
    float actual = dot_avx_512(a, b, size);

    printf(expected == actual ? "\tPASSED\n" : "\tFAILED\n");
    printf("\tExpected: %f, Actual: %f\n", expected, actual);
}
#endif

void run_tests() {
    printf("Running tests...\n");

    printf("Test 1: dot product\n");
    // test_dot();

    printf("Test 2: dot product (AVX 256)\n");
    // test_dot_avx_256();

    #ifdef __AVX512F__
    printf("Test 3: dot product (AVX 512)\n");
    test_dot_avx_512();
    #endif
}
#endif


int main() {
#ifdef TEST_MODE
    run_tests();
#else
    printf("Benchmarking...\n");

    size_t n = 1000000;

    float *a = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));

    // Initialize arrays with random values
    for (size_t i = 0; i < n; ++i) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    clock_t start, end;
    double time_taken;
    float result;

    // Benchmarking
    start = clock();

    result = dot(a, b, n);

    end = clock();

    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Dot product result: %f\n", result);
    printf("Time taken: %f seconds\n", time_taken);

    start = clock();

    result = dot_avx_256(a, b, n);

    end = clock();

    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("Dot product (AVX 256) result: %f\n", result);
    printf("Time taken: %f seconds\n", time_taken);

    free(a);
    free(b);
#endif

    return 0;
}