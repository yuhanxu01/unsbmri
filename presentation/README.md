# UNSB 论文汇报材料

本目录包含 Unpaired Neural Schrödinger Bridge (UNSB) 论文的详细汇报材料。

## 📁 文件说明

### 1. `unsb_presentation.tex`
**完整的 Beamer 演示文稿（LaTeX 源文件）**

**内容结构：**
- **第1部分：研究背景与动机**
  - 问题定义
  - Schrödinger Bridge 理论基础

- **第2部分：UNSB 算法核心思想**
  - Markov 链分解（方程7）
  - 条件生成器设计

- **第3部分：定理1详解**
  - 约束优化问题（方程9-10）
  - SB 损失和对抗约束
  - 熵正则化的深入理解
  - 转移分布（方程11-12）
  - 定理结论（方程13）

- **第4部分：代码实现细节**
  - 熵正则化的代码实现
  - 能量网络损失 (compute_E_loss)
  - SB 损失 (compute_G_loss)
  - 时间调度策略
  - OU 过程离散化
  - 网络架构
  - 训练流程

- **第5部分：配对数据策略**
  - 7种配对数据利用方法

- **第6部分：实验结果与分析**
  - MRI 图像转换结果
  - 消融实验

- **第7部分：教授可能的提问**
  - 8个常见技术问题及详细回答

**总页数：** 约35页

### 2. `FAQ_professor_questions.md`
**教授可能提问的详细 FAQ 文档**

**包含5大类问题：**

#### 熵正则化相关 (Q1.1-Q1.3)
- Q1.1: 如何计算高维空间的熵？
  - 能量网络方法
  - logsumexp 技巧
  - 代码实现详解

- Q1.2: 熵正则化的目的？
  - 理论层面：SB vs 最优传输
  - 防止模式坍塌
  - 时间衰减策略
  - 实验验证

- Q1.3: 能量网络训练的稳定性？
  - 数值问题分析
  - PyTorch 内置保护
  - 代码中的稳定措施

#### 理论基础 (Q2.1-Q2.3)
- Q2.1: 定理1的证明思路
- Q2.2: 为什么约束必须是 D_KL = 0？
- Q2.3: Markov 链分解与扩散模型的区别

#### 实现细节 (Q3.1-Q3.4)
- Q3.1: 为什么前向传播要用 detach()？
- Q3.2: 时间调度为什么用调和级数？
- Q3.3: 为什么需要两个独立样本？
- Q3.4: OU 过程的方差推导

#### 与其他方法对比 (Q4.1-Q4.3)
- Q4.1: UNSB vs CycleGAN
- Q4.2: UNSB vs DDPM 训练效率
- Q4.3: 能否用于条件生成（文本到图像）？

#### 实验与应用 (Q5.1-Q5.3)
- Q5.1: 7种配对策略哪个最有效？
- Q5.2: 为什么 batch_size=1？能否增大？
- Q5.3: 能否应用到 3D 医学图像？

**文档长度：** 约15,000字

### 3. `compile.sh`
**LaTeX 编译脚本**

自动编译 Beamer 演示文稿为 PDF 格式。

## 🚀 使用方法

### 编译演示文稿

**方法1：使用编译脚本（推荐）**
```bash
cd /home/user/unsbmri/presentation
chmod +x compile.sh
./compile.sh
```

**方法2：手动编译**
```bash
xelatex unsb_presentation.tex
xelatex unsb_presentation.tex  # 第二次生成目录
```

**输出：** `unsb_presentation.pdf`

### 查看 FAQ

```bash
# 使用 Markdown 阅读器
cat FAQ_professor_questions.md

# 或在支持 Markdown 的编辑器中打开
```

## 📋 系统要求

### LaTeX 编译
- **必需：** XeLaTeX（支持中文）
- **推荐：** TexLive Full

**Ubuntu/Debian 安装：**
```bash
sudo apt-get install texlive-full
sudo apt-get install texlive-xetex
sudo apt-get install texlive-lang-chinese
```

### 字体
- 演示文稿使用 `ctex` 宏包处理中文
- 系统需要安装中文字体（如 SimSun, SimHei）

## 🎯 汇报准备建议

### 演示文稿使用建议

**时间分配（45分钟汇报）：**
1. 背景与动机 (5分钟) - 前5页
2. UNSB 核心思想 (8分钟) - 第6-12页
3. **定理1详解 (15分钟)** - 第13-23页（重点！）
4. 代码实现 (10分钟) - 第24-31页
5. 实验结果 (5分钟) - 第32-35页
6. Q&A (2分钟预留)

**重点关注：**
- ✅ 方程9-10的物理意义
- ✅ 熵正则化的作用（必问！）
- ✅ 能量网络的实现细节
- ✅ 与 CycleGAN/DDPM 的对比

### FAQ 使用建议

**汇报前必读：**
- Q1.1, Q1.2：熵计算和目的（高频问题）
- Q2.2：KL散度约束（理论核心）
- Q3.2：时间调度（实现细节）
- Q4.1, Q4.2：方法对比（说明优势）

**打印提示卡片：**
可以将关键公式和代码片段打印在卡片上：
```
卡片1: 熵正则化公式 + logsumexp 代码
卡片2: OU 过程方差公式
卡片3: 7种配对策略对比表
```

## 📊 演示文稿特色

### 可视化元素
- ✅ TikZ 绘制的时间轴和流程图
- ✅ 彩色公式高亮（关键项用蓝色、红色标注）
- ✅ 代码块展示（真实代码片段）
- ✅ 对比表格（方法对比、消融实验）
- ✅ 算法伪代码（训练流程）

### 内容亮点
- ✅ **完整的数学推导**（方程7到方程13）
- ✅ **代码与理论对应**（每个公式都指向具体代码位置）
- ✅ **教授提问预判**（演示文稿已包含8个常见问题的回答）
- ✅ **中英文对照**（关键术语）

## 🔍 核心公式速查

### 定理1优化问题
```latex
min_φ L_SB(φ, t) = E[||x_t - x_1||²] - 2τ(1-t)H(q_φ(x_t, x_1))
s.t. D_KL(q_φ(x_1) || p(x_1)) = 0
```

### 能量网络损失
```latex
L_E = -E[E_ψ(same_pair)] + logsumexp(E_ψ(different_pairs)) + (logsumexp(...))²
```

### OU 过程更新
```latex
X_t = (1-α)X_{t-1} + α·G(X_{t-1}) + √(α(1-α)τ(1-t_{-1})) ε
```

## 📝 引用信息

**论文：**
```bibtex
@article{gushchin2023unsb,
  title={Unpaired Neural Schrödinger Bridge},
  author={Gushchin, Nikita and Kolesov, Alexander and Mokrov, Petr and Burnaev, Evgeny and Korotin, Alexander},
  journal={arXiv preprint arXiv:2305.15086},
  year={2023}
}
```

**代码库：**
- 本实现基于原始 UNSB 框架适配 MRI 图像转换任务
- 额外实现了7种配对数据利用策略

## 🆘 常见问题

### Q: 编译失败，提示找不到 ctex 包？
**A:** 安装完整的 TexLive：
```bash
sudo apt-get install texlive-lang-chinese
```

### Q: PDF 中文显示为方框？
**A:** 安装中文字体：
```bash
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
fc-cache -fv
```

### Q: 如何修改演示文稿主题？
**A:** 编辑 `unsb_presentation.tex` 第5行：
```latex
\usetheme{Madrid}  % 可改为 Berlin, Copenhagen, Warsaw 等
```

### Q: 如何导出为 PowerPoint？
**A:** 使用 pdf2pptx 工具：
```bash
# 需要安装额外软件
sudo apt-get install libreoffice
libreoffice --headless --convert-to pptx unsb_presentation.pdf
```

## 📞 联系与反馈

如果在使用过程中遇到问题或有改进建议：
1. 检查 FAQ 文档中是否有相关说明
2. 查看代码库的 issues
3. 联系论文作者或本实现的维护者

## ✨ 致谢

感谢：
- UNSB 原始论文作者
- MRI 数据集提供者
- 开源社区的 LaTeX/Beamer 模板

---

**祝汇报顺利！** 🎓

*最后更新：2025-11-17*
