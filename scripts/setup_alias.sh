#!/bin/bash

# Dynamically determine SAMVARA_DIR based on the current script location
export SAMVARA_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Function to add alias to shell config
add_alias() {
    local shell_config=$1
    if ! grep -q "alias samvara=" "$shell_config"; then
        echo "alias samvara='bash $SAMVARA_DIR/scripts/run_samvara_or_debug.sh'" >> "$shell_config"
        echo "Added samvara alias to $shell_config"
    else
        echo "samvara alias already exists in $shell_config"
    fi
}

# Add alias based on the current shell
if [ "$SHELL" = "/bin/zsh" ]; then
    add_alias ~/.zshrc
    source ~/.zshrc
else
    add_alias ~/.bashrc
    source ~/.bashrc
fi

