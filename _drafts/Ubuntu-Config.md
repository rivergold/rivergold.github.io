# :fallen_leaf:Develop Env Config

- zsh
- build-essential
- Python
- vim
- cmake
- git

## zsh

```shell
sudo apt install zsh
# Set zsh as default shell
chsh -s $(which zsh)
# Install on-my-zsh
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
# Install zsh-syntax-highlighting
cd ~/.oh-my-zsh/plugins
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ${ZSH_CUSTOM:-~/.oh-my-zsh/custom}/plugins/zsh-syntax-highlighting
```

Edit `~/.zshrc`

```shell
plugins=(
  git
  zsh-syntax-highlighting
  z)

# -----zsh
# Set *
setopt no_nomatch
# Forbiden rm -rf
alias rm='echo "rm is disabled, use trash or /bin/rm instead."'
# -----
```

---

## build-essential

```shell
sudo apt install build-essential
```

---

## Python

Install Miniconda

---
