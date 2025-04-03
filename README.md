# Optimizing Medical Image Processing: A Hybrid Approach with the FAST Framework and OpenMP

A parallel computing system to make brain imaging processing more efficient using [FAST](https://github.com/smistad/FAST/) (Framework for Heterogeneous Medical Imaging Computing and Visualization) and [OpenMP](https://www.openmp.org/wp-content/uploads/OpenMP-RefGuide-6.0-OMP60SC24-web.pdf). This project focuses on broadly utilizing all applicable system resources available on the host system for computationally intensive medical imaging processing tasks.

In our project "sequential" and "parallel" refer to the two different modes in which the image processing ***workflow*** is performed:

- **Sequential**: This implementation uses the FAST framework alone. The framework is already optimized and interally leverages OpenCL algorithms to speed up individual image processing operations (filtering, segmentation, etc.). In other words, while the low-level operations within FAST are optimized and may execute in parallel at a hardware level, the processing of multiple images is performed in a **serial** order.
- **Parallel**: In this implementation, we extend the FAST-based serial workflow by adding an additional layer of parallelism using OpenMP. Here, the processing of a batch of images is distributed amongst multiple CPU threads. So, instead of waiting for the one image to complete processing, images can be processed concurrently--squeezing out extra performance alongside FAST's internal optimizations.

1. **Import:** Loading DICOM files.
2. **Preprocessing:** Applying Intensity Normalization, Intensity Clipping, Vector Median Filtering, and Image Sharpening.
3. **Segmentation:** Using Seeded Region Growing with adaptive seed points based on image dimensions.
4. **Post-processing:** Casting image types and applying morphological operations like Dilation (and Erosion in the test pipeline).
5. **Export:** Saving the original and processed images as JPEGs.

### Initial Medical Imaging Processing Pipeline

![image](https://github.com/user-attachments/assets/6e9675cf-eccb-4523-985c-341763ced9fc)

### Revised Medical Imaging Processing Pipeline

![image](https://github.com/user-attachments/assets/85b27a61-17f8-46a0-b030-c0a6bbc28407)

## Prerequisites

- Parallel Computer
  - Computer with a multi-core processor, capable of running multiple threads of execution simultaneously
- CMake (version 3.5 or higher)
- C++ compiler with C++17 support
- FAST Framework (installed on system)
- Git

## Building the Project & Running Test Pipeline

```bash
mkdir build
cd build
cmake .. -DFAST_DIR=/opt/fast/cmake/ # default installation location, specify if otherwise
make
# Run test pipeline binary
./test_pipeline 
# other binaries include: ./img_processing_sequential or ./img_processing_parallel
```

## Project Structure

After building the target executables, and running their binaries the project root directory will resemble the following:

```
./
├── src/
│   ├── include/      # Header files (e.g., FAST directives)
│   ├── parallel/     # Parallel implementation source (main_parallel.cpp)
│   ├── sequential/   # Sequential implementation source (main_sequential.cpp)
│   └── test/         # Test pipeline source (test_pipeline.cpp)
├── build/            # Build directory
├── CMakeLists.txt    # CMake build config
├── out-parallel/     # Output from parallel processing (contains subdirectories per patient)
│   └── PGBM-XXXX/    # Example patient output directory
│       ├── *.jpg     # Original and processed image pair
├── out-sequential/   # Output from sequential processing (contains subdirectories per patient)
│   └── PGBM-XXXX/    # Example patient output directory
│       ├── *.jpg     # Original and processed image pair
├── out-test/         # Output images from the test pipeline
└── README.md         # This file
```

### Test Pipeline

- **Source**: `src/test/test_pipeline.cpp`
- **Binary**: `test_pipeline`
- **Function**: This pipeline serves as a proof of concept and prototype for the image processing pipeline. It processes a single 2D DICOM slice through the defined pipeline stages. It provides a visualization of each the processed image in each of the intermediate steps, and exports the processed image after each stage to `out-test/`.

![Test Pipeline Execution Output](https://github.com/user-attachments/assets/0e3e6881-b01a-4e08-b62d-1c38c56c6b1b)

### Sequential Image Processing (FAST)

- **Source**: `src/sequential/main_sequential.cpp`
- **Binary**: `img_processing_sequential`
- **Function**: Processes all patient T1+C (session after tumor has progressed) datasets found within the specified base data directory. It iterates through each patient folder, loads their DICOM series, and processes each sequentially. It saves the original 2D DICOM slice and the processed image pair to a patient-specific directory in `out-sequential/`.

### Parallel Image Processing (FAST + OpenMP)

- **Source**: `src/parallel/main_parallel.cpp`
- **Binary**: `img-processing_parallel`
- **Function**: Processes the same data as the above, however, the loaded DICOM images for each patient are processed in parallel batches. OpenMP is used to distribute the processing of images within a batch across multiple threads. The original/processed pair is saved to a patient-specific directory in `out-parallel/`.

## Analysis

For the purposes of this project, we needed to create and analyse data, some tools that were used include:

- **Benchmarking**: [Hyperfine](https://github.com/sharkdp/hyperfine), [time](https://linux.die.net/man/1/time)
- **Performance Analysis**: [Hotspot](https://github.com/KDAB/hotspot), [perf](https://perfwiki.github.io/main/)
- **Data Analysis**: Python, Excel

## Dataset

The dataset used for this project can be obtained from the [TCIA](https://www.cancerimagingarchive.net/collection/brain-tumor-progression/). In particular, the T1+C (T1-weighted post-contrast) subsets were used. T1+C images enhance the visualization of tumor boundaries because the contrast agent highlights areas with disrupted blood-brain barriers (common with malignant tumors), and simply provides a better contrast-to-noise ratio.

## Parallel Computer Used

**Lenovo IdeaPad 5 Pro 14ACN6**

- AMD Ryzen 7 5800U (8C, 16T)
- 16GB RAM DDR4-3200
- NVIDIA® GeForce MX450 (2GB GDDR6)

## License

This project uses the FAST framework for medical image computing and visualization.

### FAST Licensing

- The source code of FAST is licensed under the BSD 2-clause license.
- FAST binaries are linked with third-party libraries licensed under various open-source licenses, including MIT, Apache 2.0, LGPL, and others.
- For more information, see the `licenses` folder in the FAST release or refer to their [license documentation](https://github.com/smistad/FAST/blob/master/LICENSE).

Please ensure compliance with all applicable licenses when distributing or modifying this project.
