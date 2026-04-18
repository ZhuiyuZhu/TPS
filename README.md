# 🧠 TPS-AD: 中医证候智能评分系统

**TCM Pattern Scorer for Alzheimer's Disease based on Gut-Brain Axis Multi-modal Data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **科研声明**: 本系统为学术研究成果，仅供科研使用，不构成医疗诊断建议。  
> **软著申请版本**: V1.0 (2026.04) | **作者**: 朱骓羽 (西安交通大学生物强基23级编译)


## 📋 项目简介

本项目对应综述《人工智能引导下的中医药治疗阿尔茨海默病评估：证据、缺口与肠-脑轴机遇的综述》（Interdisciplinary Medicine, 投稿中），实现了文中提出的 **TCM Pattern Score (TPS)** 概念框架。

### ✨ 核心创新

1. **证候标准化引擎**：采用 BM25 信息检索算法解决中医"同证异名"问题，将自由文本映射为 5 类标准 AD 证候（肾虚髓减、痰瘀闭窍、心脾两虚、肝阳上亢、髓海不足）
2. **肠脑轴多模态融合**：整合认知评分（MMSE/MoCA）、肠道菌群代谢物（丁酸、TMAO）、神经炎症标志物（IL-1β、pTau217）与中医四诊数字化特征
3. **可解释 AI (XAI)**：基于 XGBoost-SHAP 实现透明化决策，提供个体化生物标志物贡献度分析

---

## 🚀 快速开始

### 本地安装

```bash
# 克隆仓库
git clone https://github.com/ZhuiyuZhu/TPS.git
cd TPS

# 安装依赖
pip install -r requirements.txt

# 运行 Web 界面
streamlit run tps_web_app.py
访问 http://localhost:8501 查看系统。

📁 核心文件说明

TPS/
├── tps_web_app.py              # Web 界面主程序（Streamlit）
├── syndrome_standardizer.py    # BM25 证候标准化引擎（核心算法1）
├── tps_core.py                 # XGBoost评分模型（核心算法2）
├── tps_stage2_complete.py      # 真实数据集成验证
├── standard_syndromes.json     # 证候标准库（5类AD证候定义）
├── requirements.txt            # Python 依赖列表
└── README.md                   # 本文件

🧪 数据基础
双源验证策略：
1. 
模拟数据（300例）：基于文献参数生成，含完整肠脑轴生物标志物，用于算法功能验证
2. 
真实数据（TCMEval-SDT, Nature 2025）：300例异质化中医临床记录，其中 4 例 AD/认知障碍相关病例，用于验证标准化引擎鲁棒性
数据来源：TCMEval-SDT, License: CC BY 4.0

📄 软件著作权与声明
本产品正在申请计算机软件著作权（登记中）。
版权所有 © 2026 西安交通大学生命科学与技术学院。
免责声明: 本软件按"原样"提供，仅供科研使用，不构成医疗诊断建议。作者不对因使用本软件造成的任何直接或间接损失承担责任。

📬 联系方式
 
项目负责人: 朱骓羽 (Xi'an Jiaotong University)
 
导师: 彭韵桦副教授
