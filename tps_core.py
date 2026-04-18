"""
TPS-AD: TCM Pattern Scorer for Alzheimer's Disease
基于多模态融合的中医证候智能评分系统 (软著申请版本 v0.1)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体（软著申请时图表需要中文标签）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class TPSConfig:
    """系统配置参数 - 对应综述中的标准化需求"""
    SYNDROME_TYPES = ['肾虚髓减', '痰瘀闭窍', '心脾两虚', '肝阳上亢', '髓海不足']
    GUT_BRAIN_MARKERS = ['butyrate', 'TMAO', 'LBP', 'zonulin', 'IL1β', 'TNFα', 'pTau217']
    COGNITIVE_TESTS = ['MMSE', 'MoCA', 'CDR_SB', 'ADAS_Cog']
    TCM_FEATURES = ['tongue_color_score', 'tongue_coating_score',
                    'pulse_strength', 'pulse_rhythm', 'sleep_quality',
                    'appetite_score', 'bowel_function', 'fatigue_score']
    RANDOM_STATE = 42


class MultimodalDataGenerator:
    """
    多模态数据生成器 - 基于综述 Figure 2 的概念框架
    模拟：粪便宏基因组 + 血浆标志物 + 证候量表 + 认知评分的配对数据
    """

    def __init__(self, n_samples: int = 300, missing_rate: float = 0.15):
        self.n_samples = n_samples
        self.missing_rate = missing_rate
        self.config = TPSConfig()
        np.random.seed(self.config.RANDOM_STATE)

    def generate(self) -> pd.DataFrame:
        """生成符合中医 AD 临床特征的模拟数据"""
        data = {}

        # 1. 基础人口学特征
        data['patient_id'] = [f'AD_{i:04d}' for i in range(self.n_samples)]
        data['age'] = np.random.normal(75, 8, self.n_samples).astype(int)
        data['gender'] = np.random.choice(['M', 'F'], self.n_samples)

        # ========== 修正：去掉 'mild_AD' 前面的空格 ==========
        data['disease_stage'] = np.random.choice(
            ['NC', 'MCI', 'mild_AD', 'moderate_AD'],  # ← 已修正
            self.n_samples,
            p=[0.2, 0.3, 0.35, 0.15]
        )

        # 2. 认知评分 (MMSE/MoCA 等) - 与疾病阶段相关
        base_cognitive = {
            'NC': (28, 2), 'MCI': (24, 3), 'mild_AD': (20, 4), 'moderate_AD': (14, 5)
        }
        for test in self.config.COGNITIVE_TESTS:
            scores = []
            for stage in data['disease_stage']:
                mu, sigma = base_cognitive[stage]
                scores.append(max(0, min(30, np.random.normal(mu, sigma))))
            data[test] = np.array(scores)

        # 3. 肠脑轴生物标志物 - 与 AD 病理相关
        # 短链脂肪酸 (丁酸) - 健康人群较高，AD 患者较低
        data['butyrate'] = np.where(
            np.isin(data['disease_stage'], ['NC', 'MCI']),
            np.random.normal(25, 5, self.n_samples),
            np.random.normal(15, 6, self.n_samples)
        )

        # TMAO - 炎症标志物，AD 患者升高
        data['TMAO'] = np.where(
            np.isin(data['disease_stage'], ['NC']),
            np.random.normal(3, 1, self.n_samples),
            np.random.normal(6, 2.5, self.n_samples)
        )

        # 屏障功能标志物 (LBP, Zonulin) - 通透性增加
        data['LBP'] = np.random.lognormal(2.5, 0.5, self.n_samples) + \
                      np.where(np.isin(data['disease_stage'], ['moderate_AD']), 5, 0)
        data['zonulin'] = np.random.normal(15, 4, self.n_samples) + \
                          np.where(np.isin(data['disease_stage'], ['moderate_AD']), 8, 0)

        # 炎症因子
        data['IL1β'] = np.random.exponential(2, self.n_samples) * \
                       np.where(np.isin(data['disease_stage'], ['mild_AD', 'moderate_AD']), 1.8, 1.0)
        data['TNFα'] = np.random.exponential(1.5, self.n_samples) * \
                       np.where(np.isin(data['disease_stage'], ['moderate_AD']), 2.0, 1.0)

        # 血液 AD 核心标志物
        data['pTau217'] = np.where(
            np.isin(data['disease_stage'], ['NC']),
            np.random.normal(0.5, 0.2, self.n_samples),
            np.where(
                np.isin(data['disease_stage'], ['MCI']),
                np.random.normal(1.2, 0.4, self.n_samples),
                np.random.normal(2.5, 0.8, self.n_samples)
            )
        )

        # 4. 中医证候特征 - 与生物标志物建立隐式关联
        # 肾虚髓减：与年龄、髓海不足相关，对应 pTau 升高
        kidney_deficiency = (
                (data['age'] - 65) / 20 * 0.4 +
                (data['pTau217'] - 0.5) / 2 * 0.3 +
                np.random.normal(0, 0.1, self.n_samples)
        )

        # 痰瘀闭窍：与炎症标志物正相关
        phlegm_stasis = (
                (data['IL1β'] - 2) / 4 * 0.35 +
                (data['TNFα'] - 1.5) / 3 * 0.35 +
                (data['TMAO'] - 3) / 4 * 0.2 +
                np.random.normal(0, 0.1, self.n_samples)
        )

        # 心脾两虚：与营养吸收、睡眠质量相关（对应肠道 butyrate 下降）
        heart_spleen = (
                (20 - data['butyrate']) / 20 * 0.4 +
                np.random.normal(0, 0.15, self.n_samples)
        )

        # 标准化到 0-10 的证候评分
        for syndrome, raw_score in zip(
                ['肾虚髓减', '痰瘀闭窍', '心脾两虚', '肝阳上亢', '髓海不足'],
                [kidney_deficiency, phlegm_stasis, heart_spleen,
                 np.random.normal(0.5, 0.3, self.n_samples),
                 kidney_deficiency * 0.8 + np.random.normal(0, 0.1, self.n_samples)]
        ):
            min_val, max_val = raw_score.min(), raw_score.max()
            if max_val > min_val:
                data[syndrome] = np.clip((raw_score - min_val) / (max_val - min_val) * 10, 0, 10)
            else:
                data[syndrome] = np.ones(self.n_samples) * 5

        # 5. 四诊数字化特征 - 与证候关联
        data['tongue_color_score'] = 5 + data['痰瘀闭窍'] * 0.3 + np.random.normal(0, 0.5, self.n_samples)
        data['tongue_coating_score'] = 5 + data['心脾两虚'] * 0.4 + np.random.normal(0, 0.5, self.n_samples)
        data['pulse_strength'] = 7 - data['肾虚髓减'] * 0.3 + np.random.normal(0, 0.5, self.n_samples)
        data['pulse_rhythm'] = 8 - data['心脾两虚'] * 0.2 + np.random.normal(0, 0.3, self.n_samples)
        data['sleep_quality'] = 10 - data['心脾两虚'] * 0.5 - data['肝阳上亢'] * 0.3 + np.random.normal(0, 0.5,
                                                                                                        self.n_samples)
        data['appetite_score'] = 8 - data['心脾两虚'] * 0.4 + np.random.normal(0, 0.6, self.n_samples)
        data['bowel_function'] = 8 - data['肾虚髓减'] * 0.2 + np.random.normal(0, 0.5, self.n_samples)
        data['fatigue_score'] = 3 + data['肾虚髓减'] * 0.5 + data['心脾两虚'] * 0.4 + np.random.normal(0, 0.5,
                                                                                                       self.n_samples)

        # 引入缺失值（模拟真实世界数据不完整）
        df = pd.DataFrame(data)
        for col in self.config.GUT_BRAIN_MARKERS + self.config.COGNITIVE_TESTS:
            mask = np.random.random(self.n_samples) < self.missing_rate
            df.loc[mask, col] = np.nan

        return df


class TPSFeatureEngineer:
    """
    特征工程模块：处理缺失值、跨模态对齐、标准化
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None

    def fit_transform(self, df: pd.DataFrame, config: TPSConfig) -> Tuple[np.ndarray, List[str]]:
        """构建多模态特征向量"""
        feature_cols = (
                config.COGNITIVE_TESTS +
                config.GUT_BRAIN_MARKERS +
                config.TCM_FEATURES
        )

        X = df[feature_cols].copy()
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)

        self.feature_names = feature_cols
        return X_scaled, self.feature_names


class ExplainableTPSModel:
    """
    可解释 TPS 评分模型 - 使用 XGBoost + SHAP
    """

    def __init__(self):
        self.models = {}
        self.explainers = {}
        self.config = TPSConfig()

    def fit(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], feature_names: List[str]):
        """训练多任务模型"""
        for syndrome in self.config.SYNDROME_TYPES:
            print(f"训练证候模型: {syndrome}")

            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.config.RANDOM_STATE,
                objective='reg:squarederror'
            )

            model.fit(X, y_dict[syndrome])
            self.models[syndrome] = model
            explainer = shap.TreeExplainer(model)
            self.explainers[syndrome] = explainer

        return self

    def predict_score(self, X: np.ndarray) -> pd.DataFrame:
        """预测 TPS 各维度得分"""
        predictions = {}
        for syndrome, model in self.models.items():
            predictions[syndrome] = model.predict(X)
        return pd.DataFrame(predictions)

    def explain_instance(self, X_instance: np.ndarray, feature_names: List[str],
                         syndrome: str = '痰瘀闭窍') -> Dict:
        """个体化解释"""
        if syndrome not in self.explainers:
            raise ValueError(f"未找到证候 {syndrome} 的解释器")

        shap_values = self.explainers[syndrome].shap_values(X_instance.reshape(1, -1))
        feature_importance = dict(zip(feature_names, shap_values[0]))
        top_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        return {
            'syndrome': syndrome,
            'predicted_score': self.models[syndrome].predict(X_instance.reshape(1, -1))[0],
            'top_drivers': top_features,
            'interpretation': self._generate_interpretation_text(top_features)
        }

    def _generate_interpretation_text(self, top_features: List[Tuple[str, float]]) -> str:
        """生成自然语言解释"""
        explanations = []
        for feature, value in top_features[:3]:
            direction = "升高" if value > 0 else "降低"
            if feature in ['IL1β', 'TNFα', 'TMAO', 'LBP']:
                explanations.append(f"炎症/肠脑轴标志物 {feature} 的{direction}（贡献值: {value:.2f}）")
            elif feature in ['MMSE', 'MoCA']:
                explanations.append(f"认知评分 {feature} 的{direction}（贡献值: {value:.2f}）")
            else:
                explanations.append(f"临床特征 {feature} 的{direction}（贡献值: {value:.2f}）")
        return "；".join(explanations)


class TPSVisualization:
    """TPS 可视化模块"""

    @staticmethod
    def plot_syndrome_radar(scores: pd.Series, patient_id: str, save_path: Optional[str] = None):
        """生成证候雷达图"""
        categories = scores.index.tolist()
        values = scores.values.tolist()
        values += values[:1]

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label='TPS 评分', color='#2E86AB')
        ax.fill(angles, values, alpha=0.25, color='#2E86AB')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 10)
        ax.set_title(f'患者 {patient_id} 的 TCM Pattern Score (TPS)', fontsize=14, pad=20)
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"已保存雷达图: {save_path}")
        plt.close()


# 主运行流程
def main_demo():
    """TPS 系统演示流程"""
    print("=" * 60)
    print("TPS-AD: 中医证候智能评分系统 v0.1")
    print("基于肠脑轴多模态数据的可解释 AI 模型")
    print("=" * 60)

    # 1. 数据生成
    print("\n[步骤 1] 生成多模态模拟数据 (n=300)...")
    generator = MultimodalDataGenerator(n_samples=300, missing_rate=0.15)
    df = generator.generate()
    print(f"数据维度: {df.shape}")
    print(f"证候分布:\n{df[TPSConfig.SYNDROME_TYPES].mean().round(2)}")

    # 2. 特征工程
    print("\n[步骤 2] 多模态特征工程 (处理缺失值/标准化)...")
    engineer = TPSFeatureEngineer()
    X, feature_names = engineer.fit_transform(df, TPSConfig())
    print(f"特征矩阵维度: {X.shape}")
    print(f"特征列表: {feature_names}")

    # 3. 准备标签
    y_dict = {syndrome: df[syndrome].values for syndrome in TPSConfig.SYNDROME_TYPES}

    # 4. 训练模型
    print("\n[步骤 3] 训练可解释 TPS 模型 (XGBoost + SHAP)...")
    tps_model = ExplainableTPSModel()
    tps_model.fit(X, y_dict, feature_names)

    # 5. 预测与解释
    print("\n[步骤 4] 生成 TPS 评分与个体化解释...")
    predictions = tps_model.predict_score(X[:5])

    for i in range(3):
        patient_id = df.iloc[i]['patient_id']
        print(f"\n--- 患者 {patient_id} ---")
        print(f"疾病阶段: {df.iloc[i]['disease_stage']}")
        print(f"MMSE: {df.iloc[i]['MMSE']:.1f}, MoCA: {df.iloc[i]['MoCA']:.1f}")
        print(f"pTau217: {df.iloc[i]['pTau217']:.2f}")

        # TPS 评分
        scores = predictions.iloc[i]
        print("TPS 证候评分:")
        for syndrome in TPSConfig.SYNDROME_TYPES:
            print(f"  {syndrome}: {scores[syndrome]:.1f}/10")

        # 解释性分析
        explanation = tps_model.explain_instance(X[i], feature_names, '痰瘀闭窍')
        print(f"\n[可解释性分析 - 痰瘀闭窍]")
        print(f"预测评分: {explanation['predicted_score']:.1f}")
        print(f"关键驱动因素: {explanation['interpretation']}")

        # 生成可视化
        TPSVisualization.plot_syndrome_radar(
            scores, patient_id,
            save_path=f'tps_radar_{patient_id}.png'
        )

    print("\n" + "=" * 60)
    print("演示完成。已生成 TPS 雷达图 (tps_radar_*.png)")
    print("软著申请要点：")
    print("1. 多模态数据融合（中医+认知+肠脑轴）")
    print("2. 可解释 AI (SHAP 值展示特征贡献)")
    print("3. 标准化证候评分输出 (0-10 量表)")
    print("=" * 60)


if __name__ == "__main__":
    main_demo()
