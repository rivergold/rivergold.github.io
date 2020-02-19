# Install
## Ubuntu
Unreal for linux can be installed on Ubuntu 16.04, other version may not success. The followings show a breif tips for installation, for much more details please go to [Unreal Engine 4 Documentation](https://docs.unrealengine.com/latest/INT/Platforms/Linux/BeginnerLinuxDeveloper/SettingUpAnUnrealWorkflow/index.html).
1. Registering for an Epic Games account
2. Download the source code of UE4 from github
3. Run
    - `bash Setup.sh`
    - `bash GenerateProjectFiles.sh`
    - `make -j4`

<br>

## Problems and Solutions
- **Error** `loginit warning, logexit warning, logdatabase error" Refusing to run with the root privileges`
    It means that UE4 cannot be maked by `root` or `sudo`, use a non-root user and just run `make -j4` withour `sudo`

    ***References:***
    - [UE4 Anserhub: Build errors](https://answers.unrealengine.com/questions/316725/i-dont-know-what-this-build-error-means.html)


<br>

*** 

<br>
