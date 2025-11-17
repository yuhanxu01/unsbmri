#!/bin/bash

# UNSB Presentation Compilation Script

echo "====================================="
echo "编译 UNSB Beamer 演示文稿"
echo "====================================="

# 检查 pdflatex 是否安装
if ! command -v xelatex &> /dev/null
then
    echo "错误: xelatex 未安装"
    echo "请安装 TexLive: sudo apt-get install texlive-full"
    exit 1
fi

# 编译 Beamer 演示文稿
echo ""
echo "[1/3] 第一次编译..."
xelatex -interaction=nonstopmode unsb_presentation.tex > /dev/null 2>&1

echo "[2/3] 第二次编译（生成目录）..."
xelatex -interaction=nonstopmode unsb_presentation.tex > /dev/null 2>&1

echo "[3/3] 第三次编译（更新引用）..."
xelatex -interaction=nonstopmode unsb_presentation.tex

# 检查编译是否成功
if [ -f "unsb_presentation.pdf" ]; then
    echo ""
    echo "✓ 编译成功！"
    echo "  输出文件: unsb_presentation.pdf"

    # 清理辅助文件
    echo ""
    echo "清理辅助文件..."
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb

    echo ""
    echo "====================================="
    echo "完成！可以打开 PDF 查看演示文稿"
    echo "====================================="
else
    echo ""
    echo "✗ 编译失败，请检查错误信息"
    echo "查看日志文件: unsb_presentation.log"
    exit 1
fi
