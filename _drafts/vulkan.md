# Basics

- [Vulkan Home](https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html)

# Install

- vulkan driver (Nivida, AMD, Intel)

## Ubuntu

### Install graphics driver

Intel graphics driver only support windows. So we use mesa-vulkan-driver on Linux.

- Check driver

```bash
lshw -c video
modinfo i915
```

Ref [stackoverflow: How can I find what video driver is in use on my system?](https://askubuntu.com/questions/23238/how-can-i-find-what-video-driver-is-in-use-on-my-system)

Install

```bash
sudo add-apt-repository ppa:oibaf/graphics-drivers
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-utils
```

Test

```bash
vulkaninfo | less
```

Ref [LinuxConfig: Install And Test Vulkan On Linux](https://linuxconfig.org/install-and-test-vulkan-on-linux)

Ref [stackoverflow: How to upgrade Intel Graphics driver?](https://askubuntu.com/questions/1065852/how-to-upgrade-intel-graphics-driver)

Ref [stackoverflow: How to install intel graphics driver for using vulkan on Ubuntu 16.04](https://askubuntu.com/questions/807857/how-to-install-intel-graphics-driver-for-using-vulkan-on-ubuntu-16-04)

***References:***

- [Khronos: Conformant Products](https://www.khronos.org/conformance/adopters/conformant-products)
- [01.org: INTEL® GRAPHICS FOR LINUX*](https://01.org/zh/node/18011)
    - INDUSTRY-LEADING OPEN SOURCE GRAPHICS DRIVERS