# Brain Imaging Segmentation using the FAST Framework

A parallel computing system for medical diagnoses with image matching using the [FAST](https://github.com/smistad/FAST/) (Framework for Heterogeneous Medical Imaging Computing and Visualization) framework. This project focuses on broadly utilizing all applicable system resources available on the host system for computationally intensive medical imaging tasks involving brain tumor detection.

## Prerequisites

- CMake (version 3.16 or higher)
- C++ compiler with C++17 support
- FAST Framework (installed on system)
- Git

## Building the Project

```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Project Structure

```
.
├── src/            # Source files
├── CMakeLists.txt  # CMake configuration
└── README.md       # This file
```

## License

This project uses the FAST framework for medical image computing and visualization.

### FAST Licensing

- The source code of FAST is licensed under the BSD 2-clause license.
- FAST binaries are linked with third-party libraries licensed under various open-source licenses, including MIT, Apache 2.0, LGPL, and others.
- For more information, see the `licenses` folder in the FAST release or refer to their [license documentation](https://github.com/smistad/FAST/blob/master/LICENSE).

Please ensure compliance with all applicable licenses when distributing or modifying this project.
