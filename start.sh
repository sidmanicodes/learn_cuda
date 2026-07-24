#!/bin/bash
set -e

mkdir -p ~/.ssh
chmod 700 ~/.ssh

mkdir -p /run/sshd

# sshd rebuilds its session environment via PAM instead of inheriting the
# container's ENV, so PATH/LD_LIBRARY_PATH/NVIDIA_* GPU vars would otherwise
# be invisible over SSH. Writing them to /etc/environment fixes that.
env > /etc/environment

if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
fi

ssh-keygen -A
exec /usr/sbin/sshd -D
