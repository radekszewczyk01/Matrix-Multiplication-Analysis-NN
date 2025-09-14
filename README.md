# Complete Development Environment Setup Guide for C/C++/CUDA on WSL (Ubuntu)

This guide provides all the necessary steps to set up a complete C, C++, OpenCL, and CUDA development environment on the Windows Subsystem for Linux (WSL) using an Ubuntu distribution.

## Prerequisites

Before you begin, ensure you have the following:
1.  A Windows 10 or 11 machine with an NVIDIA GPU (RTX series recommended).
2.  The latest NVIDIA drivers installed on your **Windows** host system.
3.  WSL 2 installed with an active Ubuntu distribution.

---

## Step 1: Install Core Build Tools and Git

These are the fundamental packages for any C/C++ development. `build-essential` includes `gcc`, `g++`, and `make`.

Open your Ubuntu (WSL) terminal and run the following commands:

```bash
# Update the package list
sudo apt update

# Install C/C++ compilers, make, and Git
sudo apt install build-essential git
````

-----

## Step 2: Install OpenBLAS for High-Performance CPU Computations

To compile code that uses BLAS functions (like `cblas_sgemm`), you need the OpenBLAS development library. This will provide the necessary header files (e.g., `cblas.h`) and the compiled library.

```bash
sudo apt install libopenblas-dev
```

-----

## Step 3: Install the OpenCL Environment

The OpenCL environment consists of a few parts: the generic headers, the driver manager (ICD loader), and the specific hardware drivers.

```bash
# Install the generic OpenCL headers needed for compilation
sudo apt install opencl-headers

# Install the OpenCL Installable Client Driver (ICD) loader development package
# This allows the system to find and manage different OpenCL implementations (NVIDIA, Intel, etc.)
sudo apt install ocl-icd-opencl-dev

# (Optional) Install the Intel OpenCL driver if you also want to use your integrated GPU
sudo apt install intel-opencl-icd

# Install a tool to verify the OpenCL setup
sudo apt install clinfo
```

-----

## Step 4: Install the NVIDIA CUDA Toolkit

This is the most crucial step for NVIDIA GPU programming. It provides the `nvcc` compiler, the `cuBLAS` library, and the NVIDIA OpenCL implementation. We will use the network installer method, which is robust and always provides the latest version.

1.  **Run the following commands in your WSL terminal to add the NVIDIA repository and install the toolkit:**

    ```bash
    # Download and install the NVIDIA repository keyring to authenticate packages
    wget [https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb)
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    # Clean up the downloaded file
    sudo rm cuda-keyring_1.1-1_all.deb

    # Update the package list to include the new NVIDIA repository
    sudo apt-get update

    # Install the latest stable CUDA toolkit meta-package
    sudo apt-get install -y cuda-toolkit
    ```

2.  **Configure Environment Variables**

    After the installation is complete, you must tell your terminal where to find the CUDA tools. You do this by editing the `.bashrc` file.

      * Open the file with the `nano` text editor:
        ```bash
        nano ~/.bashrc
        ```
      * Scroll to the very bottom of the file and add the following two lines. Using `/usr/local/cuda` is recommended as it's a symbolic link that always points to the latest installed CUDA version.
        ```bash
        # Add CUDA to the environment PATH
        export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
        export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
        ```
      * Save the file and exit `nano` by pressing `Ctrl+O`, then `Enter`, then `Ctrl+X`.
      * Apply the changes to your current terminal session by running:
        ```bash
        source ~/.bashrc
        ```
        (Alternatively, just close and reopen the terminal).

-----

## Step 5: Final Verification

Your environment should now be fully configured. You can verify each component by running the following commands in a new WSL terminal.

  * **Check GCC:**

    ```bash
    gcc --version
    ```

  * **Check OpenCL platforms:**

    ```bash
    clinfo
    ```

    *(You should see at least one platform, "NVIDIA CUDA", listed).*

  * **Check CUDA Compiler (nvcc):**

    ```bash
    nvcc --version
    ```

    *(This should display the version of the installed CUDA toolkit).*

You are now ready to compile and run the project.




