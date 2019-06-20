# Vulkan

- [Vulkan](https://www.khronos.org/vulkan/)
- [Vulkan SDK](https://vulkan.lunarg.com/)

# :seedling: Good Books & Blogs

- [Vulkan Docs](https://vulkan.lunarg.com/doc/sdk/latest/linux/getting_started.html)

- [Glumes Blog: 进击的 Vulkan 移动开发（一）之今生前世](https://glumes.com/post/vulkan/vulkan-tutorial-concept/)

- [Vulkan in 30 minutes](https://renderdoc.org/vulkan-in-30-minutes.html)

# :seedling: Example

- [Github: LunarG/VulkanSamples](https://github.com/LunarG/VulkanSamples)

# :seedling: Concept

## Validation Layer

用来做调试用的，在release开发完的代码时，不需要使用Validation Layers

<p align="center">
  <img
  src="https://upload-images.jianshu.io/upload_images/9890707-ef14b2b207e2f883.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240" width="30%">
</p>

> Vulkan validation layers contain a set of libraries which help find potential problems in created applications. Their debugging capabilities include, but are not limited to, validating parameters passed to Vulkan functions, validating texture and render target formats, tracking Vulkan objects and their lifetime and usage, and checking for potential memory leaks or dumping (displaying/printing) Vulkan API function calls.

## Extention

扩展，丰富vulkan的功能。例如：用于跨平台渲染。

- Instance Extention
- Device Extensions

***Ref:*** [Glumes: 进击的 Vulkan 移动开发之 Instance & Device & Queue](https://juejin.im/post/5c32c3d6e51d45524b029e26)

## Instance

A Vulkan Instance is an object that gathers the state of an application.

It encloses information such as an application name, name and version of an engine used to create an application, or enabled instance-level extensions and layers.

## Physical Devices and Logical Devices

TBD

## Queues

> A queue's responsibility is to gather the jobs (command buffers) and dispatch them to the physical device for processing.

> **A queue is the abstracted mechanism used to submit commands to the hardware.** You will see later how a Vulkan application builds up a command buffer full of commands and then submits them onto a queue for asynchronous processing by the GPU hardware.

Each of the physical devices advertises one or more queues. These queues are categorized into different families, where each family has very specific functionalities.

**在创建`physical device`之后，在创建`device`之前，需要构建queue**

## Image & Buffer

我目前的理解，image和buffer都是对一段内存的表示，具体的不同还有待学习。

***Ref:*** [知乎专栏-黑客与画家: 一分钟学 Vulkan 之 Image & Buffer](https://zhuanlan.zhihu.com/p/20758451)

***References:***

- [知乎专栏-黑客与画家: Texture 是什么？](https://zhuanlan.zhihu.com/p/26393153)

## Command buffer & command pool

向commonad buffer中注册需要在GPU上运行的command

为了管理的方面，采用command buffer pool统一对command buffer进行管理。针对于不同类型的queue family，需要配置不同的command buffer pool。

> Effectively, you would have to create a command buffer pool for each unique queue family that the application intends to use. Since you specified only one queue family when you created the device, one command buffer pool suffices for these samples.

> Finishing the command buffer recording does not make the GPU do anything. In order for the GPU to process a command buffer, you must submit it onto the GPU's queue using vkQueueSubmit().

## Render Pass

是一个过程的集合

- [ ] 不是很懂

***References:***

- [知乎专栏-一起学Vulkan图形开发: 入门：调试Vulkan Samples（七）——Render Pass](https://zhuanlan.zhihu.com/p/47769196)

- [博客园: Vulkan Tutorial 13 Render passes](https://www.cnblogs.com/heitao/p/6979494.html)

## Layout

- Descriptor Set Layout
- Pipeline Layout

目前的理解：Layout使用对配置的描述

<!--  -->
<br>

***

<br>
<!--  -->

# :seedling: Install

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

**Test**

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

## macOS

Ref [Lunarg Doc macOS: Getting Started with the Vulkan SDK](https://vulkan.lunarg.com/doc/view/1.0.69.0/mac/getting_started.html)

***References:***

- [Khronos Forum Vulkan: Why does my vulkan instance have no any validation layers?](https://community.khronos.org/t/why-does-my-vulkan-instance-have-no-any-validation-layers/7007)

<!--  -->
<br>

***

<br>
<!--  -->

# :seedling: Function

## `xxxInfo` and `xxxProperties`

- `xxxInfo`: paramters feed into vulkan sdk
- `xxxProperties`: paramters get from vulkan sdk

```c++
uint32_t extensions_count = 0;
VkResult result = VK_SUCCESS;
result = vkEnumerateInstanceExtensionPropertie(nullptr, &extensions_count, nullptr);
if ((result != VK_SUCCESS) || (extensions_count== 0)) {
  std::cout << "Could not get the number ofInstance extensions." << std::endl;
  return false;
}
```

<!--  -->
<br>

***

<br>
<!--  -->

# :seedling: Program

## Step

1. Init `Instance`
2. 

## Enable Validation Layer

There are two methods to enable validation layer:

- Use environment (recommend)
- Enable in source code

```bash
export VK_INSTANCE_LAYERS=VK_LAYER_LUNARG_api_dump:VK_LAYER_LUNARG _core_validation
```

<!--  -->
<br>

***

<br>
<!--  -->

# :seedling: Awesome Tools

## OpenVX

***References:*** 

- [知乎: OpenCV和OpenVX有什么联系和区别？](https://www.zhihu.com/question/37894914)

# TBD

## Vulkan Compute

- [Phoronix: A Simple Dive Into Vulkan Compute Programming](https://www.phoronix.com/scan.php?page=news_item&px=Vulkan-Compute-Small-Sample)
- [Github Glavnokoman: Vulkan Compute Example](https://github.com/Glavnokoman/vulkan-compute-example)

- [Descriptor Set Layouts and Pipeline Layouts](https://vulkan.lunarg.com/doc/view/1.0.26.0/linux/tutorial/html/08-init_pipeline_layout.html)