#!/bin/bash

tag=2023-03-13
docker run -p 10000:8888 jupyter/minimal-notebook:${tag}
