# Use the official Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Add the deadsnakes PPA to get Python 3.8
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa

# Update package list and install necessary packages including Python 3.8, psycopg2, and LLVM distutils
RUN apt-get update && \
    apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-psycopg2 \
    python3.8-distutils \
    pkg-config \
    libssl-dev \
    build-essential \
    gdb \
    curl \
    git \
    vim \
    gcc \
    make \
    libreadline-dev \
    zlib1g-dev \
    libicu-dev \
    pkg-config \
    libclang-dev \
    llvm-dev \
    clang \
    flex \
    bison \
    wget \
    unzip \
    sudo \
    cmake \
    && apt-get clean

# Download torchlib CPU version and extract it to /home
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.1%2Bcpu.zip
RUN unzip libtorch-cxx11-abi-shared-with-deps-2.3.1+cpu.zip -d /home

# Set Python 3.8 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Set root password
RUN echo "root:rootpassword" | chpasswd

# Create a non-root user 'postgres' with an empty password and add to sudoers
RUN useradd -m -s /bin/bash postgres && \
    passwd -d postgres && \
    echo "postgres ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install Rust using rustup and cargo-pgrx as the postgres user
# Ensure rustc 1.78.0 rustc --version
# Ensure cargo-pgrx v0.11.4 cargo-pgrx --version
USER postgres
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    echo 'source $HOME/.cargo/env' >> $HOME/.bashrc && \
    /bin/bash -c "source $HOME/.cargo/env && rustup install 1.78.0" && \
    /bin/bash -c "source $HOME/.cargo/env && cargo install cargo-pgrx --version '0.11.4' --locked"

# Switch back to root to set system-wide environment variables
USER root

# Set the path for Rust and Cargo for both root and postgres users
ENV PATH="/home/postgres/.cargo/bin:${PATH}"
RUN echo 'export PATH="$PATH:/home/postgres/.cargo/bin"' >> /etc/profile
RUN echo 'export PATH="$PATH:/home/postgres/.cargo/bin"' >> /home/postgres/.bashrc

# Set environment variables for both root and other users
ENV PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig"
ENV LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
ENV NEURDBPATH="/code/neurdb-dev"
RUN echo "export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}" >> /etc/profile && \
    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> /etc/profile && \
    echo "export NEURDBPATH=${NEURDBPATH}" >> /etc/profile && \
    echo "export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}" >> /home/postgres/.bashrc && \
    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> /home/postgres/.bashrc && \
    echo "export NEURDBPATH=${NEURDBPATH}" >> /home/postgres/.bashrc && \
    echo 'export LIBCLANG_PATH=$(llvm-config --libdir)' >> /etc/profile && \
    echo 'export LIBCLANG_PATH=$(llvm-config --libdir)' >> /home/postgres/.bashrc

# Set the working directory
WORKDIR /code/neurdb-dev

# Copy the build script into the container
COPY init.sh /usr/local/bin/init.sh

# Ensure the build script is executable
RUN chmod +x /usr/local/bin/init.sh

# Expose port
EXPOSE 5432

# Change to non-root user 'postgres' for database compilation and running the init script
USER postgres
# Command to run the init.sh script
CMD ["bash", "/usr/local/bin/init.sh"]
