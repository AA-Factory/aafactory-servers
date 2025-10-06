# aafactory-servers

This repository contains the official servers for the AAFactory platform.

## Overview

Each server in this repository is designed to be easily plugged into the AAFactory tool, enabling flexible and modular usage. These servers are distributed as portable and isolated containers, which means you can run them either locally on your machine or remotely on any compatible infrastructure.

## Architecture & Benefits

- **Portability & Isolation:**  
  Servers are packaged as containers, ensuring consistent behavior across different environments and simplifying deployment.

- **Flexible Deployment:**  
  You can run servers locally for development or testing, or deploy them remotely for production use.

- **Worker Model:**  
  Each server acts as a worker that connects to a central Redis instance. When a request is placed in Redis, the appropriate worker (regardless of its physical location) can independently pick up and process the task.

- **Scalability:**  
  This architecture allows you to scale horizontally by running multiple workers across different machines or environments, all coordinated via Redis.

## Usage

1. Deploy one or more servers (workers) from this repository.
2. Connect them to your Redis instance.
3. Plug the servers into the AAFactory tool.
4. When a request is made, the correct worker will pick up the task and process it, independent of where it is running.


## Deployment
- push a tag with the name of the server and its version. E.g: zonos-{{version}}
- this will trigger a build that will be saved in https://hub.docker.com/u/aafactory