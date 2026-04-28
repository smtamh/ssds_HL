FROM osrf/ros:humble-desktop

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

# Basic tools + editors
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano \
    vim \
    curl \
    wget \
    lsb-release \
    gnupg \
    ca-certificates \
    less \
    && rm -rf /var/lib/apt/lists/*

# Additional ROS 2 packages + Poco/gtest/gmock + MuJoCo/OSQP build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-ament-cmake-clang-format \
    ros-humble-xacro \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-robot-state-publisher \
    ros-humble-tf2-tools \
    ros-humble-rviz2 \
    ros-humble-v4l2-camera \
    ros-dev-tools \
    libpoco-dev \
    libgtest-dev \
    libgmock-dev \
    build-essential \
    pkg-config \
    cmake \
    git \
    libgl1-mesa-dev \
    libx11-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    && rm -rf /var/lib/apt/lists/*

# Gazebo Ignition Fortress repository + install
RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
      -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends ignition-fortress && \
    rm -rf /var/lib/apt/lists/*

# robotpkg repository for Pinocchio
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
      -o /etc/apt/keyrings/robotpkg.asc && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
      > /etc/apt/sources.list.d/robotpkg.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends "robotpkg-py3*-pinocchio" && \
    rm -rf /var/lib/apt/lists/*

# uv install
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# libfranka install
RUN VERSION=0.19.0 && CODENAME=jammy && \
    curl -LO https://github.com/frankarobotics/libfranka/releases/download/${VERSION}/libfranka_${VERSION}_${CODENAME}_amd64.deb && \
    dpkg -i libfranka_${VERSION}_${CODENAME}_amd64.deb && \
    rm -f libfranka_${VERSION}_${CODENAME}_amd64.deb

# MuJoCo build and install
# mujoco_ros_hardware vendors a simulate.cc that still uses mjWARN_VGEOMFULL,
# which was removed in MuJoCo 3.6. Pin below that breaking API change.
ARG MUJOCO_VERSION=3.5.0
RUN cd /root && \
    git clone --branch ${MUJOCO_VERSION} --depth 1 https://github.com/google-deepmind/mujoco.git && \
    cd mujoco && \
    mkdir build && cd build && \
    cmake .. && \
    cmake --build . && \
    cmake --install . && \
    cd /root && \
    rm -rf /root/mujoco

# OSQP build and install
RUN cd /root && \
    git clone https://github.com/osqp/osqp.git && \
    cd osqp && \
    mkdir build && cd build && \
    cmake -G "Unix Makefiles" .. && \
    cmake --build . && \
    cmake --build . --target install && \
    cd /root && \
    rm -rf /root/osqp

# OSQP-Eigen build and install
RUN cd /root && \
    git clone https://github.com/gbionics/osqp-eigen.git && \
    cd osqp-eigen && \
    mkdir build && cd build && \
    cmake .. && \
    make && \
    make install && \
    cd /root && \
    rm -rf /root/osqp-eigen

# Environment
RUN cat <<'EOF' >> /root/.bashrc
source /opt/ros/humble/setup.bash
[ -f /root/ros2_ws/install/setup.bash ] && source /root/ros2_ws/install/setup.bash
source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
export PATH=/root/.local/bin:/opt/openrobots/bin:$PATH
export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/opt/openrobots/lib:/usr/local/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH
export CMAKE_PREFIX_PATH=/opt/openrobots:/usr/local:$CMAKE_PREFIX_PATH
export ROS2_WS=/root/ros2_ws

uv-docker() {
    UV_PROJECT_ENVIRONMENT=/root/ssds_HL/.venv/310 uv "$@"
}
EOF


WORKDIR /root
CMD ["/bin/bash"]
