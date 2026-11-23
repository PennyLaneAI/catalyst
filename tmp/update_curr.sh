#!/bin/bash
git fetch
git merge origin/$(git branch --show-current)
./tmp/push_curr.sh
