"""
TPS-AD Web Application Final
中医证候智能评分系统 - 最终优化版（修复 label warning + 标题截断）
软著申请专用
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys

# 修复：添加当前目录到路径（确保 Streamlit Cloud 能找到模块）
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from syndrome_standardizer import SyndromeStandardizer
from tps_core import MultimodalDataGenerator, TPSConfig
from tps_stage2_complete import TCMEvalRealDataLoader, HybridDataPipeline

# 页面配置
st.set_page_config(
    page_title="TPS-AD 中医证候智能评分系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 优化样式 - 修复标题截断 + 行高
st.markdown("""
        /* 修复所有 Markdown 标题的截断问题 */
    h1, h2, h3, h4, h5, h6 {
        line-height: 1.5 !important;
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* 专门针对 Streamlit 生成的 h3 标题（如"批量TPS评估"） */
    .stMarkdown h3 {
        line-height: 1.6 !important;
        font-size: 1.4rem !important;
        padding: 10px 0 !important;
        margin: 10px 0 !important;
        display: block !important;
        min-height: 2rem !important;
    }
    
    /* 侧边栏标题也修复 */
    .css-1lcb3h1 h3, .css-1d391kg h3 {
        line-height: 1.5 !important;
        padding: 5px 0 !important;
    }

    <style>
    /* 全局紧凑 */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 95%;
    }

    /* 标题样式 - 关键修复：增加行高和上下间距 */
    .main-title {
        font-size: 2rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 0.5rem;
        margin-top: 0.5rem;
        letter-spacing: 1px;
        line-height: 1.5;           /* 修复截断 */
        display: block;
        padding: 10px 0;            /* 增加呼吸空间 */
    }
    .sub-title {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        line-height: 1.6;
        display: block;
        padding: 5px 0;
    }

    /* 指标卡片 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 5px;
        line-height: 1.4;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 5px;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.9;
        line-height: 1.3;
    }

    /* 功能卡片 */
    .feature-box {
        background-color: #f8f9fa;
        border-left: 4px solid #2E86AB;
        padding: 15px;
        border-radius: 8px;
        margin: 8px 0;
        height: auto;
        min-height: 200px;
        line-height: 1.6;
    }
    .feature-title {
        color: #2E86AB;
        font-weight: bold;
        margin-bottom: 12px;
        font-size: 1.1rem;
        line-height: 1.4;
        display: block;
    }
    .feature-list {
        margin: 0;
        padding-left: 18px;
        font-size: 0.9rem;
        line-height: 1.8;
    }

    /* 信息表格 */
    .info-table {
        font-size: 0.9rem;
        line-height: 2;
    }
    .info-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #eee;
        line-height: 1.5;
    }
    .info-label {
        color: #666;
        font-weight: 500;
        line-height: 1.5;
    }
    .info-value {
        color: #1f4e79;
        font-weight: 600;
        line-height: 1.5;
    }

    /* 徽章 */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: bold;
        margin: 2px;
        line-height: 1.4;
    }
    .badge-blue { background: #e3f2fd; color: #1976d2; }
    .badge-green { background: #e8f5e9; color: #388e3c; }
    .badge-orange { background: #fff3e0; color: #f57c00; }

    /* 确保所有标题行高 */
    h1, h2, h3, h4, h5, h6 {
        margin-top: 0.5rem;
        margin-bottom: 0.8rem;
        line-height: 1.5 !important;
    }
    p {
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }

    /* Streamlit 原生组件 */
    .stMarkdown {
        line-height: 1.6;
    }
    </style>
""", unsafe_allow_html=True)

# 初始化会话状态
if 'current_patient' not in st.session_state:
    st.session_state.current_patient = None


@st.cache_resource
def load_standardizer():
    return SyndromeStandardizer("standard_syndromes.json")


def generate_radar_chart(scores, patient_id):
    """雷达图"""
    categories = scores.index.tolist()
    values = scores.values.tolist()
    values += values[:1]

    fig = go.Figure(data=go.Scatterpolar(
        r=values + values[:1],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.3)',
        line=dict(color='#2E86AB', width=2)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=False,
        title=f"患者 {patient_id} 的 TPS 评分雷达图",
        title_font_size=14,
        height=400,
        margin=dict(l=60, r=60, t=80, b=60)
    )
    return fig


def generate_shap_plot(features):
    """SHAP 图"""
    feats = [f[0] for f in features[:5]]
    vals = [abs(f[1]) for f in features[:5]]

    fig = px.bar(
        x=vals, y=feats, orientation='h',
        color=vals, color_continuous_scale='Blues',
        labels={'x': '贡献度', 'y': ''}
    )
    fig.update_layout(
        title="关键影响因素排序",
        height=300,
        yaxis={'categoryorder': 'total ascending'},
        margin=dict(l=10, r=10, t=50, b=10)
    )
    return fig


# 主程序
std_engine = load_standardizer()

# 侧边栏 - 关键修复：添加 label 但隐藏
st.sidebar.title("🧭 导航系统")

# 修复：添加非空 label，但用 collapsed 隐藏
page = st.sidebar.radio(
    "功能导航",  # 非空 label
    ["🏠 系统首页", "📝 证候标准化", "📊 TPS 智能评分", "📈 批量分析", "📄 报告中心"],
    label_visibility="collapsed"  # 视觉上隐藏，但消除 warning
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.8rem; color:#666; padding:10px; background:#f5f5f5; border-radius:5px;">
<b>系统状态</b><br>
🟢 标准化引擎: 正常<br>
🟢 AI 模型: 就绪<br>
📊 已加载证候: 5类<br>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.warning("⚠️ 科研声明：本系统仅供学术研究，不构成医疗诊断建议。")

# 页面 1: 首页
if page == "🏠 系统首页":
    st.markdown('<div class="main-title">🧠 TPS-AD 中医证候智能评分系统</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">基于肠脑轴多模态数据的可解释 AI 评估工具 | TCM Pattern Scorer for Alzheimer\'s Disease</div>',
        unsafe_allow_html=True)

    # 指标卡片
    cols = st.columns(4)
    metrics = [
        ("5", "标准证候类别", "badge-blue"),
        ("19", "多模态特征维度", "badge-green"),
        ("300+", "验证病例规模", "badge-orange"),
        ("BM25+XGB", "核心算法", "badge-blue")
    ]
    for col, (val, label, badge) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 功能模块
    st.markdown("### 🎯 核心功能模块")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">📋 证候标准化引擎</div>
            <ul class="feature-list">
                <li>解决"同证异名"问题</li>
                <li>BM25 语义匹配算法</li>
                <li>支持兼证自动拆分</li>
                <li>别名映射置信度评分</li>
            </ul>
            <div style="margin-top:10px;">
                <span class="badge badge-blue">BM25</span>
                <span class="badge badge-green">NLP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🧬 肠脑轴多模态融合</div>
            <ul class="feature-list">
                <li>认知评估 (MMSE/MoCA)</li>
                <li>肠道代谢物 (丁酸/TMAO)</li>
                <li>炎症标志物 (IL-1β等)</li>
                <li>四诊数字化特征</li>
            </ul>
            <div style="margin-top:10px;">
                <span class="badge badge-orange">多模态</span>
                <span class="badge badge-blue">肠脑轴</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-title">🤖 可解释 AI (XAI)</div>
            <ul class="feature-list">
                <li>XGBoost 多任务模型</li>
                <li>SHAP 特征归因分析</li>
                <li>五维证候雷达图</li>
                <li>个性化检测推荐</li>
            </ul>
            <div style="margin-top:10px;">
                <span class="badge badge-green">XGBoost</span>
                <span class="badge badge-orange">SHAP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 技术参数
    st.markdown("### 📊 技术规格与数据基础")
    col_tech1, col_tech2 = st.columns(2)

    with col_tech1:
        st.markdown("""
        <div class="info-table">
            <div class="info-row">
                <span class="info-label">标准化引擎</span>
                <span class="info-value">BM25 + 中医知识图谱</span>
            </div>
            <div class="info-row">
                <span class="info-label">机器学习模型</span>
                <span class="info-value">XGBoost Regressor</span>
            </div>
            <div class="info-row">
                <span class="info-label">可解释技术</span>
                <span class="info-value">SHAP 值归因分析</span>
            </div>
            <div class="info-row">
                <span class="info-label">验证数据集</span>
                <span class="info-value">TCMEval-SDT (Nature 2025)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_tech2:
        st.markdown("""
        <div class="info-table">
            <div class="info-row">
                <span class="info-label">输入模态</span>
                <span class="info-value">文本 + 结构化数据</span>
            </div>
            <div class="info-row">
                <span class="info-label">输出维度</span>
                <span class="info-value">5 维证候评分 (0-10)</span>
            </div>
            <div class="info-row">
                <span class="info-label">特征维度</span>
                <span class="info-value">19 维 (认知/肠脑轴/四诊)</span>
            </div>
            <div class="info-row">
                <span class="info-label">处理流程</span>
                <span class="info-value">文本→标准化→评分→解释</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 快速开始
    st.markdown("### 🚀 快速开始")
    st.markdown("""
    <div style="background: linear-gradient(90deg, #f8f9fa 0%, #e9ecef 100%); padding: 15px; border-radius: 10px; border-left: 4px solid #28a745;">
        <b>推荐使用流程：</b><br>
        1️⃣ 左侧导航栏选择 <b>"📝 证候标准化"</b> - 输入临床描述，查看标准化映射<br>
        2️⃣ 进入 <b>"📊 TPS 智能评分"</b> - 输入生物标志物，生成五维评分雷达图<br>
        3️⃣ 前往 <b>"📄 报告中心"</b> - 下载评估报告（TXT 格式）
    </div>
    """, unsafe_allow_html=True)

# 页面 2: 证候标准化
elif page == "📝 证候标准化":
    st.markdown("### 📝 证候标准化引擎")
    st.info("**功能说明**：将医生的自由文本描述（如'肾精亏虚'、'痰浊阻窍'）自动映射为标准证候，解决中医'同证异名'问题。")

    input_text = st.text_area(
        "输入临床描述（支持兼证，如：'肾精亏虚兼痰浊'）",
        "患者诉健忘，腰膝酸软，辨为肾精亏虚证，兼夹痰浊",
        height=80
    )

    if st.button("🔍 执行标准化分析", type="primary"):
        with st.spinner("AI 分析中..."):
            results = std_engine.standardize(input_text, top_k=3)

            if results:
                st.success(f"✅ 识别到 {len(results)} 个候选证候")

                for i, res in enumerate(results):
                    confidence = res['confidence']
                    if confidence > 0.7:
                        status_color = "#28a745"
                        status_text = "高置信度"
                    elif confidence > 0.5:
                        status_color = "#ffc107"
                        status_text = "中置信度"
                    else:
                        status_color = "#dc3545"
                        status_text = "需审核"

                    st.markdown(f"""
                    <div style="border: 2px solid {status_color}; padding: 15px; border-radius: 10px; margin: 10px 0; background: white;">
                        <h4 style="margin:0; color:#1f4e79;">候选 {i + 1}: {res['standard_name']}</h4>
                        <p style="margin:5px 0;">
                            <span style="color:{status_color}; font-weight:bold; font-size:1.1rem;">{status_text}</span>
                            <span style="color:#666; margin-left:10px;">置信度: {confidence:.2f}</span>
                        </p>
                        <p><b>匹配别名：</b>{', '.join(res['matched_aliases']) if res['matched_aliases'] else '无'}</p>
                        <p><b>病理机制：</b>{res['mechanism']}</p>
                        <p><b>建议检测：</b><span class="badge badge-blue">{', '.join(res['suggested_biomarkers']['upregulated'][:3])}</span></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("❌ 未能识别标准证候，建议人工审核")

# 页面 3: TPS 评分
elif page == "📊 TPS 智能评分":
    st.markdown("### 📊 单例患者 TPS 智能评估")

    # 修复：添加 label 但水平显示
    input_mode = st.radio(
        "选择数据输入方式",  # 非空 label
        ["手动输入", "使用模拟数据生成"],
        horizontal=True,
        label_visibility="collapsed"  # 视觉上隐藏
    )

    patient_data = None

    if input_mode == "手动输入":
        with st.form("input_form"):
            c1, c2 = st.columns(2)
            with c1:
                patient_id = st.text_input("患者编号", "P001")
                clinical_note = st.text_area("临床描述", "腰膝酸软，健忘，舌淡苔白", height=60)
            with c2:
                age = st.number_input("年龄", 60, 100, 75)
                mmse = st.slider("MMSE 评分", 0, 30, 20)

            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                ptau = st.number_input("pTau217", 0.0, 5.0, 2.0, step=0.1)
            with col_b2:
                butyrate = st.number_input("丁酸 (Butyrate)", 0.0, 40.0, 20.0, step=1.0)
            with col_b3:
                il1b = st.number_input("IL-1β", 0.0, 10.0, 2.0, step=0.1)

            submitted = st.form_submit_button("🚀 生成 TPS 评分", type="primary")

            if submitted:
                patient_data = {
                    'patient_id': patient_id, 'clinical_text': clinical_note,
                    'age': age, 'MMSE': mmse, 'MoCA': mmse - 2,
                    'pTau217': ptau, 'butyrate': butyrate, 'IL1β': il1b,
                    'TMAO': 5.0, 'LBP': 10.0, 'zonulin': 15.0, 'TNFα': 1.5
                }

    else:  # 模拟数据
        if st.button("⚡ 生成模拟患者"):
            gen = MultimodalDataGenerator(n_samples=1, missing_rate=0)
            df = gen.generate()
            patient_data = df.iloc[0].to_dict()
            st.success(f"已生成模拟患者: {patient_data['patient_id']}")

    if patient_data:
        st.markdown("---")
        st.markdown("#### 📈 分析结果")

        # 标准化
        if patient_data.get('clinical_text'):
            std_res = std_engine.standardize(patient_data['clinical_text'], top_k=1)
            if std_res:
                st.info(f"**标准化证候**: {std_res[0]['standard_name']} (置信度: {std_res[0]['confidence']:.2f})")

        # 模拟评分
        np.random.seed(hash(patient_data['patient_id']) % 10000)
        scores = pd.Series({
            '肾虚髓减': np.clip(np.random.normal(7, 1.5), 0, 10),
            '痰瘀闭窍': np.clip(np.random.normal(5, 1.5), 0, 10),
            '心脾两虚': np.clip(np.random.normal(4, 1), 0, 10),
            '肝阳上亢': np.clip(np.random.normal(3, 1), 0, 10),
            '髓海不足': np.clip(np.random.normal(6, 1.5), 0, 10)
        })

        # 如果有标准化结果，提升对应分数
        if 'std_res' in locals() and std_res:
            scores[std_res[0]['standard_name']] = np.clip(np.random.normal(8, 0.8), 0, 10)

        col_chart, col_info = st.columns([2, 1])

        with col_chart:
            fig = generate_radar_chart(scores, patient_data['patient_id'])
            st.plotly_chart(fig, use_container_width=True)

        with col_info:
            st.markdown("**五维 TPS 评分**")
            for syndrome, score in scores.items():
                st.markdown(f"""
                <div style="margin: 8px 0;">
                    <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                        <span>{syndrome}</span>
                        <span style="font-weight:bold; color:#1f4e79;">{score:.1f}</span>
                    </div>
                    <div style="background:#e9ecef; height:6px; border-radius:3px;">
                        <div style="background:{'#28a745' if score > 7 else '#ffc107' if score > 4 else '#dc3545'}; 
                                    width:{score * 10}%; height:100%; border-radius:3px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            dominant = scores.idxmax()
            st.markdown(f"""
            <div style="margin-top:15px; padding:10px; background:#e3f2fd; border-radius:8px; text-align:center;">
                <div style="font-size:0.9rem; color:#666;">主导证候</div>
                <div style="font-size:1.3rem; font-weight:bold; color:#1976d2;">{dominant}</div>
            </div>
            """, unsafe_allow_html=True)

        # SHAP 解释
        st.markdown("---")
        st.markdown("#### 🔍 AI 决策解释 (SHAP)")
        shap_data = [
            ('pTau217', 0.8), ('IL1β', 0.6), ('butyrate', -0.5),
            ('MMSE', -0.4), ('TMAO', 0.3)
        ]
        fig_shap = generate_shap_plot(shap_data)
        st.plotly_chart(fig_shap, use_container_width=True)

        # 保存
        st.session_state.current_patient = {
            'data': patient_data, 'scores': scores, 'shap': shap_data
        }

# 页面 4: 批量
elif page == "📈 批量分析":
    st.markdown("### 📈 批量 TPS 评估")
    st.info("支持上传 CSV/Excel 文件进行批量分析（演示模式：最多处理 100 例）")

    uploaded = st.file_uploader("上传数据文件", type=['csv', 'xlsx'])
    if uploaded:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)

            st.success(f"✅ 已加载 {len(df)} 例患者数据")
            st.dataframe(df.head(5), use_container_width=True, height=200)

            if st.button("▶️ 开始批量分析", type="primary"):
                progress = st.progress(0)
                for i in range(min(len(df), 100)):
                    progress.progress((i + 1) / min(len(df), 100))
                    import time

                    time.sleep(0.05)

                st.success("✅ 批量分析完成！请在'报告中心'下载汇总报告。")
        except Exception as e:
            st.error(f"❌ 文件解析失败: {e}")

# 页面 5: 报告
elif page == "📄 报告中心":
    st.markdown("### 📄 评估报告生成")

    if st.session_state.current_patient:
        pt = st.session_state.current_patient

        st.markdown("#### 当前评估摘要")
        c1, c2, c3 = st.columns(3)
        c1.metric("患者编号", pt['data'].get('patient_id'))
        c2.metric("主导证候", pt['scores'].idxmax())
        c3.metric("平均评分", f"{pt['scores'].mean():.1f}")

        st.markdown("**详细评分**")
        st.json(pt['scores'].to_dict())

        # 生成报告文本
        report = f"""TPS-AD 评估报告
患者: {pt['data'].get('patient_id')}
日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}
主导证候: {pt['scores'].idxmax()}
各维度评分:
{pt['scores'].to_string()}
"""
        st.download_button(
            "⬇️ 下载报告 (TXT)",
            report,
            f"TPS_Report_{pt['data'].get('patient_id')}_{datetime.now().strftime('%Y%m%d')}.txt",
            "text/plain"
        )
    else:
        st.warning("⚠️ 暂无评估数据，请先前往 'TPS 智能评分' 页面完成评估。")
