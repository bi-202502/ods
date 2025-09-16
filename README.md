# ODS - Object Detection System

A machine learning system for text classification organized as a uv workspace monorepo.

## Architecture

This project is organized into two main workspaces:

- **`api/`** - REST API service for machine learning inference and training
- **`web/`** - Web UI for interacting with the machine learning models

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) following the official instructions.

## Setup

Install all workspace dependencies from the project root:

```bash
uv sync
```

This will install dependencies for both the API and web workspaces.
