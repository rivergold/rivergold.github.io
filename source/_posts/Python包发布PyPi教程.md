---
title: Python包发布PyPi教程
categories: Tech
tags:
  - Python
abbrlink: 7c8dcfaa
date: 2021-02-01 21:49:22
---


rivergold最近开发了几个包，希望将它们到PyPi从而可以通过pip一键安装，总结了一份教程share给大家。

## 核心步骤

总结起来主要分为5个步骤，分别为：

1. Write your package code and test it.
2. Write `setup.py`
3. Build wheel
4. Config `pypirc`
5. Upload into PyPI

下面以rivergold写的`rutils`包（一个集合了日常图像开发常用的小工具的包）作为样例，依次对以上步骤详细介绍下。

## Step-1 Write your package

You need to manage your code as a Python package. Here is an example:

```shell
rutils
├── CHANGELOG.md
├── LICENSE
├── README.md
├── rutils
│   ├── common.py
│   ├── __init__.py
│   ├── time.py
│   └── video.py
├── setup.py
└── test
    ├── common
    │   ├── test_run_command.py
    │   └── test_str2path.py
    └── test_video.py

```

You need to add `__init__.py` to each subfolder to make your folder as a Python Package. And this structure will also help you import module conveniently. You can get source code from [GitHub](https://github.com/rivergold/rutils).

## Step-2 Write `setup.py`

You can use `setup.py` to declare the package information, author, dependences and other metadate about this package.

Here is an `setup.py` example:

```python
from pathlib import Path
from setuptools import setup, find_packages

this_dir = Path(__file__).parent.resolve()
readme_path = this_dir / 'README.md'
with readme_path.open('r', encoding='utf-8') as f:
    long_description = f.read()

setup(name='rutils',
      version='0.2.0',
      packages=find_packages(),
      description='Utils for Computer Vision',
      install_requires=['pillow', 'pymediainfo', 'rlogger'],
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='rivergold',
      author_email='jinghe.rivergold@gmail.com',
      license='MIT')
```

## Step-3 Build wheel

Run `python setup.up bdist_wheel` will build a wheel named as `<package_name>-<version>-py3-none-any.whl` in `./dist`. This wheel will be uploaded into PyPI in next step.

## Step-4 Config `pypirc`

`pypirc` is a config file for pypi repo url. The file path is `~/.pypirc`.

PyPI has two repo, one is formal, another is test:

- Formal: `https://pypi.org/simple/`
- Test: `https://test.pypi.org/simple`

You'd better upload your package into test repo to do a test at first time, and upload to formal repo after testing ok.

Here is an `pypirc` example with multi url repo:

```shell
[distutils]
index-servers =
    testpypi
    pypi

[testpypi]
repository: https://test.pypi.org/legacy/
username: xxx
password: xxx

[pypi]
repository: https://upload.pypi.org/legacy/
username: xxx
password: xxx
```

Upload package

```python
twine upload -r <repo_name_in_pypirc> dist/*
```

[Blog: pypirc 中 distutils 的多服务器配置中的默认服务器](https://www.coder.work/article/1269078)

## Step-5 Upload to PyPI

We use `twine` to upload package into PyPI.

```shell
# Test repo
twine upload -r testpypi dist/*
# pip install from test repo
pip install <package name> -i https://test.pypi.org/simple

# Formal repo
twine upload -r pypi dist/*
# pip install from formal repo
pip install <package name> -i https://pypi.org/simple
```

注：上传到 PyPI formal repo 的 package，需要等待一段时间后，才可以使用`pip install <package_name>`（没有`-i`）进行安装

---

## Tricks

### Add Project Description

You need to add followings into `setup.py`

```python
# Read git repo README.md
this_dir = Path(__file__).parent.resolve()
readme_path = this_dir / 'README.md'
with readme_path.open('r', encoding='utf-8') as f:
    long_description = f.read()

setup(name='rutils',
      version='0.2.0',
      packages=find_packages(),
      description='Utils for Computer Vision',
      install_requires=['pillow', 'pymediainfo', 'rlogger'],
      long_description=long_description,             # Add this line
      long_description_content_type='text/markdown', # Add this line
      author='rivergold',
      author_email='jinghe.rivergold@gmail.com',
      license='MIT')
```

[PyPA: Making a PyPI-friendly README](https://packaging.python.org/guides/making-a-pypi-friendly-readme/)

### Package Version

注: pypi 不允许版本号的覆盖，所以每次 upload 的版本号都要不一样才行；所以最好先用 Test repo 做测试，测试 ok 后再上传到 Formal repo

[PEP 440 -- Version Identification and Dependency Specification](https://www.python.org/dev/peps/pep-0440/#version-specifiers)
