FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    g++ \
    make \
    cmake \
    curl \
    ca-certificates \
    bzip2 \
    bash \
    git \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*


# 使用 TARGETARCH 而不是 dpkg
ARG TARGETARCH

# 与 verifiers/v1/harnesses/node.py 保持一致，预装 Node.js 22.19.0。
# node.py 会检查 /var/tmp/vf-node/bin/node 的精确版本，匹配后跳过运行时下载。
ARG VF_NODE_VERSION=22.19.0
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        NODE_ARCH="x64"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        NODE_ARCH="arm64"; \
    else \
        echo "Unsupported architecture: $TARGETARCH" >&2; exit 1; \
    fi && \
    mkdir -p /var/tmp/vf-node && \
    curl --fail --show-error --location --retry 5 \
        "https://nodejs.org/dist/v${VF_NODE_VERSION}/node-v${VF_NODE_VERSION}-linux-${NODE_ARCH}.tar.gz" \
        | tar -xz -C /var/tmp/vf-node --strip-components=1 && \
    /var/tmp/vf-node/bin/node --version | grep -Fx "v${VF_NODE_VERSION}"

# The ACP harnesses use the exact Node installation above.  Keep it on PATH so
# npm-based harness installations below and runtime subprocesses use the same
# Node/npm pair as verifiers/v1/harnesses/node.py.
ENV PATH=/var/tmp/vf-node/bin:$PATH

RUN if [ "$TARGETARCH" = "amd64" ]; then \
        MINIFORGE_ARCH="x86_64"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        MINIFORGE_ARCH="aarch64"; \
    else \
        echo "Unsupported architecture: $TARGETARCH" && exit 1; \
    fi && \
    curl -L "https://github.com/conda-forge/miniforge/releases/download/26.3.2-3/Miniforge3-Linux-${MINIFORGE_ARCH}.sh" -o /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/miniforge && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/miniforge/bin:$PATH

# Install uv once in the image.  runtimes/base.py will find this copy instead
# of downloading uv for every PEP 723 harness script.
RUN curl --fail --show-error --location --retry 5 https://astral.sh/uv/install.sh \
        | UV_INSTALL_DIR=/usr/local/bin sh \
    && uv --version

# Pre-install dependencies used by the shared Bash/Null chat program and ACP
# runner. Hermes is installed separately from its own locked source tree.
RUN uv pip install --system \
    openai \
    agent-client-protocol==0.12.1 \
    mcp==2.0.0 \
    'httpx[socks]' \
    httpx2 \
    tenacity \
    certifi \
    math-verify

# Hermes is installed by the harness from a GitHub tarball when this directory
# is absent.  Put the exact source tree and its locked virtualenv in the image
# so Hermes setup is filesystem-only at rollout time.  Keep this version in
# sync with verifiers/v1/harnesses/hermes_agent/harness.py.
ARG VF_HERMES_VERSION=v2026.9.14
ENV VF_HERMES_DIR=/var/tmp/vf-hermes-agent-${VF_HERMES_VERSION}
RUN mkdir -p "$VF_HERMES_DIR" && \
    curl --fail --show-error --location --retry 5 \
        "https://github.com/NousResearch/hermes-agent/archive/refs/tags/${VF_HERMES_VERSION}.tar.gz" \
        | tar -xz --strip-components=1 -C "$VF_HERMES_DIR" && \
    uv sync --project "$VF_HERMES_DIR" --locked --no-dev --extra acp --extra mcp && \
    touch "$VF_HERMES_DIR/.ready"

# The Bash/Null chat program and ACP runner are PEP 723 scripts.  Match their
# dependency metadata in launch.py and runner.py so a fresh runtime can resolve
# both environments from the image's uv cache without downloading packages.
# GSM8K uses the same cache for its verifier script.
RUN printf '%s\n' \
        '# /// script' \
        '# requires-python = ">=3.10,<3.15"' \
        '# dependencies = ["agent-client-protocol==0.12.1", "httpx"]' \
        '# ///' > /tmp/vf-acp-runner.py && \
    printf '%s\n' \
        '# /// script' \
        '# requires-python = ">=3.10"' \
        '# dependencies = ["openai", "mcp==2.0.0", "httpx", "httpx2", "tenacity", "certifi"]' \
        '# ///' > /tmp/vf-chat-program.py && \
    printf '%s\n' \
        '# /// script' \
        '# dependencies = ["math-verify"]' \
        '# ///' > /tmp/vf-gsm8k-verify.py
RUN uv sync --script /tmp/vf-acp-runner.py --no-config && \
    uv sync --script /tmp/vf-chat-program.py --no-config && \
    uv sync --script /tmp/vf-gsm8k-verify.py --no-config && \
    rm -f /tmp/vf-acp-runner.py /tmp/vf-chat-program.py /tmp/vf-gsm8k-verify.py

# Pre-install the npm ACP adapters used by claude_code, codex, and pi.  The
# paths and versions mirror their harness.py constants so ensure_installed()
# observes the existing files and does not download them during a rollout.
ARG VF_CLAUDE_CODE_VERSION=2.1.278
ARG VF_CLAUDE_ACP_VERSION=0.79.0
ARG VF_CODEX_VERSION=0.155.1
ARG VF_CODEX_ACP_VERSION=1.12.0
ARG VF_PI_VERSION=0.86.1
ARG VF_PI_MCP_VERSION=2.34.0
ARG VF_PI_ACP_VERSION=0.0.33

RUN set -eux; \
    # claude code
    claude_dir="/var/tmp/vf-claude-agent-acp-${VF_CLAUDE_CODE_VERSION}-${VF_CLAUDE_ACP_VERSION}"; \
    claude_packages="${claude_dir}/packages"; \
    mkdir -p "${claude_packages}"; \
    npm install --prefix "${claude_packages}" --no-audit --no-fund --omit=dev \
        "@anthropic-ai/claude-code@${VF_CLAUDE_CODE_VERSION}" \
        "@agentclientprotocol/claude-agent-acp@${VF_CLAUDE_ACP_VERSION}" >/dev/null; \
    touch "${claude_dir}/.ready"; \
    # codex
    codex_dir="/var/tmp/vf-codex-${VF_CODEX_VERSION}-${VF_CODEX_ACP_VERSION}"; \
    codex_packages="${codex_dir}/acp"; \
    mkdir -p "${codex_packages}"; \
    npm install --prefix "${codex_packages}" --ignore-scripts --no-audit --no-fund --omit=dev \
        "@agentclientprotocol/codex-acp@${VF_CODEX_ACP_VERSION}" \
        "@openai/codex@${VF_CODEX_VERSION}" >/dev/null; \
    touch "${codex_dir}/.ready"; \
    # pi
    pi_packages=/var/tmp/vf-pi/mcp; \
    mkdir -p "${pi_packages}"; \
    npm install --prefix "${pi_packages}" --ignore-scripts --no-audit --no-fund --omit=dev \
        "@earendil-works/pi-coding-agent@${VF_PI_VERSION}" \
        "pi-mcp-adapter@${VF_PI_MCP_VERSION}" \
        "pi-acp@${VF_PI_ACP_VERSION}" >/dev/null; \
    printf '%s' "${VF_PI_VERSION}:${VF_PI_MCP_VERSION}:${VF_PI_ACP_VERSION}" > "${pi_packages}/.versions"

WORKDIR /

CMD ["/bin/bash"]
