# Parallelizing Brain Image Processing using the FAST Framework and OpenMP

A parallel computing system for medical diagnoses with image matching using [FAST](https://github.com/smistad/FAST/) (Framework for Heterogeneous Medical Imaging Computing and Visualization). This project focuses on broadly utilizing all applicable system resources available on the host system for computationally intensive medical imaging processing tasks.

This requires first implementing a sequential implementation of an image processing pipeline, and then adapting a further parallelized version (since FAST already provides some level of heterogeneous compute in their filtering/segmentation algorithms). The parallelized version is defined by the use of the OpenMP API which utilizes an implementation of multithreading making it more efficient, in this case, to process multiple DICOM images.

### Initially Proposed Medical Imaging Pipeline
![image](https://github.com/user-attachments/assets/5e62af4c-539f-4fb5-b001-0179b4682789)
## Prerequisites

- CMake (version 3.16 or higher)
- C++ compiler with C++17 support
- FAST Framework (installed on system)
- Git

## Building the Project

```bash
mkdir build
cd build
cmake .. -DFAST_DIR=/opt/fast/cmake/ # default installation location, varies if built from source
cmake --build . --config Release
# Run Executable (test pipeline example)
./test_pipeline 
```

## Project Structure

After building the target executables, the project root directory will resemble the following:

```
./
├── src/
├── build/
├── CMakeLists.txt
├── out-parallel/ # exported images (to .jpg) after running ./img_processing_parallel
├── out-sequential/ # exported images (to .jpg) after running ./img_processing_sequential 
├── out-test/ # exported images (to .jpg) after running ./test-pipeline
└── README.md
```

### Test Pipeline

**Implemented in /src/test/test-pipeline.cpp** -- this file outputs .jpg images depicting the transformations applied to a single 2D DICOM file throughout the image processing pipeline. This file is meant to provide a structured procedure of processing a medical image, before advancing to processing a set of DICOM images in an MRI scan session. When the its executable is run, visualization of the stages of processing are also shown.

### Sequential Image Processing

**Implemented in /src/test/main_sequential.cpp** -- this file processes an entire directory filled with 23 DICOM images corresponding to one full brain scan. Here, they are processed sequentially and each image goes through the processing pipeline -- where the original, and final processed image are saved to out-sequential/.

### Parallel Image Processing

**Implemented at /src/test/main_parallel.cpp** -- WIP

## Dataset

The dataset used for this project can be obtained from the [TCIA](https://www.cancerimagingarchive.net/collection/brain-tumor-progression/). In particular, the T1+C (T1-weighted post-contrast) subsets were used. T1+C images enhance the visualization of tumor boundaries because the contrast agent highlights areas with disrupted blood-brain barriers (common with malignant tumors), and simply provides a better contrast-to-noise ratio.

## License

This project uses the FAST framework for medical image computing and visualization.

### FAST Licensing

- The source code of FAST is licensed under the BSD 2-clause license.
- FAST binaries are linked with third-party libraries licensed under various open-source licenses, including MIT, Apache 2.0, LGPL, and others.
- For more information, see the `licenses` folder in the FAST release or refer to their [license documentation](https://github.com/smistad/FAST/blob/master/LICENSE).

Please ensure compliance with all applicable licenses when distributing or modifying this project.
