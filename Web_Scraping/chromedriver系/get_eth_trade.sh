#!/bin/bash
# Set version to 2.7.10
export PATH=~/.pyenv/shims:$PATH
#export PYENV_VERSION=anadonda3-5.2.0

# DO STUFF
python ./get_eth_trade.py
# python --version # ==> Python 2.7.10

# Reset version
#unset PYENV_VERSION