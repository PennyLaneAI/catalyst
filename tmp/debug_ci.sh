#!/bin/bash
git add .
git commit --allow-empty -m "Debug CI"
git push origin $(git branch --show-current)
