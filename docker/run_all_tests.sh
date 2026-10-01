#!/bin/bash
# ============================================================
# 一键测试脚本 — 在 Docker 容器中运行所有测试文件
# 用法:
#   cd docker
#   ./run_all_tests.sh              # 使用默认镜像
#   ./run_all_tests.sh v1.4.2       # 指定版本号
#   ./run_all_tests.sh vf-harness-prewarm:local  # 测试本地构建的镜像
#   ./run_all_tests.sh v1.4.0 --pull  # 强制重新拉取镜像
# ============================================================

set -e

# --- 配置 ---
IMAGE_NAME="alexshan2docker/optimization-env"
IMAGE_VERSION="${1:-v1.4.0}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_VERSION}"
if [[ "${1:-}" == *:* ]]; then
    FULL_IMAGE="$1"
fi
DO_PULL="${2}"

# 脚本所在目录（test/）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 测试文件列表
TEST_FILES=(
    "test/agents.py"
)

TOTAL_TESTS=${#TEST_FILES[@]}
PASSED=0
FAILED=0
RESULTS=()

# --- 函数 ---
print_separator() {
    echo -e "${BLUE}============================================================${NC}"
}

print_header() {
    echo ""
    print_separator
    echo -e "${BOLD}${BLUE}  Docker 优化环境一键测试${NC}"
    echo -e "${BLUE}  镜像: ${FULL_IMAGE}${NC}"
    echo -e "${BLUE}  测试文件数: ${TOTAL_TESTS}${NC}"
    print_separator
    echo ""
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}错误: 未找到 Docker，请先安装 Docker${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Docker 已安装${NC}"
}

pull_image() {
    # 如果指定了 --pull 或者本地没有镜像，则拉取
    if [[ "$DO_PULL" == "--pull" ]] || ! docker image inspect "$FULL_IMAGE" &> /dev/null; then
        echo -e "${YELLOW}拉取镜像 ${FULL_IMAGE} ...${NC}"
        docker pull "$FULL_IMAGE"
        echo -e "${GREEN}✓ 镜像拉取完成${NC}"
    else
        echo -e "${GREEN}✓ 使用本地镜像 ${FULL_IMAGE}${NC}"
    fi
}

run_single_test() {
    local test_file=$1
    local test_name="${test_file%.py}"

    echo ""
    print_separator
    echo -e "${BOLD}▶ 运行测试: ${test_name} (${test_file})${NC}"
    print_separator

    # 在容器中运行测试文件
    local exit_code=0
    if docker run --rm --network none \
        -v "${SCRIPT_DIR}/${test_file}:/tmp/run_test.py:ro" \
        "${FULL_IMAGE}" \
        uv run --no-project --offline --python /opt/miniforge/bin/python /tmp/run_test.py; then
        exit_code=0
    else
        exit_code=$?
    fi

    if [[ $exit_code -eq 0 ]]; then
        echo ""
        echo -e "${GREEN}✓ ${test_name} — 通过${NC}"
        RESULTS+=("${GREEN}✓${NC} ${test_name}")
        PASSED=$((PASSED + 1))
    else
        echo ""
        echo -e "${RED}✗ ${test_name} — 失败 (退出码: ${exit_code})${NC}"
        RESULTS+=("${RED}✗${NC} ${test_name}")
        FAILED=$((FAILED + 1))
    fi
}

print_summary() {
    echo ""
    echo ""
    print_separator
    echo -e "${BOLD}${BLUE}  测试结果汇总${NC}"
    print_separator

    for result in "${RESULTS[@]}"; do
        echo -e "  ${result}"
    done

    echo ""
    echo -e "  总计: ${TOTAL_TESTS}  |  ${GREEN}通过: ${PASSED}${NC}  |  ${RED}失败: ${FAILED}${NC}"

    print_separator

    if [[ $FAILED -eq 0 ]]; then
        echo -e "${GREEN}${BOLD}✓ 全部测试通过！${NC}"
        echo ""
        exit 0
    else
        echo -e "${RED}${BOLD}✗ ${FAILED} 个测试失败${NC}"
        echo ""
        exit 1
    fi
}

# --- 主流程 ---
main() {
    print_header
    check_docker
    pull_image

    for test_file in "${TEST_FILES[@]}"; do
        if [[ -f "${SCRIPT_DIR}/${test_file}" ]]; then
            run_single_test "$test_file"
        else
            echo -e "${RED}警告: 测试文件不存在 — ${test_file}${NC}"
            RESULTS+=("${RED}✗${NC} ${test_file} (文件不存在)")
            FAILED=$((FAILED + 1))
        fi
    done

    print_summary
}

main
