# ===============================
# 🌱 PATH / ENVIRONMENT SETUP
# ===============================

# Homebrew (optional: Mac環境ではbrewがない可能性あり)
if command -v brew &> /dev/null; then
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# PostgreSQL
export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"

# nvm
export NVM_DIR="$HOME/.nvm"
if [ -s "/opt/homebrew/opt/nvm/nvm.sh" ]; then
  . "/opt/homebrew/opt/nvm/nvm.sh"
fi
if [ -s "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm" ]; then
  . "/opt/homebrew/opt/nvm/etc/bash_completion.d/nvm"
fi

# ===============================
# ⚙️ SHELL SETTINGS
# ===============================

# bash補完が有効なら読み込み
if [ -f /etc/bash_completion ]; then
  . /etc/bash_completion
fi

# 履歴関連
HISTCONTROL=ignoredups:erasedups
HISTSIZE=10000
SAVEHIST=10000
shopt -s histappend
shopt -s cmdhist
shopt -s checkwinsize

# ===============================
# 🧰 ALIASES
# ===============================
alias ll='ls -lh --color=auto'
alias la='ls -lha --color=auto'
alias grep='grep --color=auto'

# ===============================
# 🎨 PROMPT (bash対応・Git付き)
# ===============================

# 色定義
RED="\[\033[0;31m\]"
GREEN="\[\033[0;32m\]"
YELLOW="\[\033[0;33m\]"
BLUE="\[\033[0;34m\]"
PURPLE="\[\033[0;35m\]"
CYAN="\[\033[0;36m\]"
RESET="\[\033[0m\]"

# Gitプロンプト設定
if [ -f /usr/share/git/completion/git-prompt.sh ]; then
  source /usr/share/git/completion/git-prompt.sh
elif [ -f ~/.git-prompt.sh ]; then
  source ~/.git-prompt.sh
fi

export GIT_PS1_SHOWDIRTYSTATE=1
export GIT_PS1_SHOWCOLORHINTS=1
export GIT_PS1_SHOWUNTRACKEDFILES=1

# プロンプト2行構成（bash対応）
export PS1="${BLUE}\u${RESET}@${GREEN}\h${RESET}:${YELLOW}\w${RESET}${PURPLE}\$(__git_ps1 ' (%s)')${RESET}\$ "

# ===============================
# 🧩 WELCOME MESSAGE
# ===============================
echo -e "${CYAN}Welcome, $(whoami)!${RESET}  $(date '+%Y-%m-%d %H:%M:%S')"