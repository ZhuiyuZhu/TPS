"""
TPS-AD 集成演示：非标准化输入 → 标准证候 → TPS 评分 → 生物标志物推荐
软著申请核心流程展示
"""

from tps_core import MultimodalDataGenerator, TPSFeatureEngineer, ExplainableTPSModel, TPSConfig, TPSVisualization
from syndrome_standardizer import SyndromeStandardizer
import pandas as pd
import numpy as np


def integrated_pipeline():
    """完整流程：从自由文本到 TPS 评分"""
    print("=" * 80)
    print("TPS-AD 集成演示：证候标准化 + 多模态评分")
    print("=" * 80)

    # Step 1: 初始化证候标准化引擎
    print("\n【步骤 1】初始化证候标准化引擎...")
    std_engine = SyndromeStandardizer("standard_syndromes.json")

    # Step 2: 模拟医生输入（ messy data ）
    clinical_notes = [
        "患者诉健忘，腰膝酸软，辨为肾精亏虚证",
        "头重如裹，舌苔厚腻，当属痰浊阻窍",
        "心烦失眠，脉细数，考虑心脾两虚"
    ]

    # Step 3: 标准化并生成模拟数据
    print("\n【步骤 2】证候标准化 → 生成多模态数据...")
    standardized_patients = []

    for note in clinical_notes:
        std_results = std_engine.standardize(note, top_k=1)
        if std_results:
            primary_syndrome = std_results[0]['standard_name']
            confidence = std_results[0]['confidence']

            # 根据证候类型生成对应的生物标志物特征（模拟真实临床）
            patient_data = generate_patient_by_syndrome(primary_syndrome, confidence)
            patient_data['patient_id'] = f"STD_{len(standardized_patients):04d}"
            patient_data['raw_note'] = note
            patient_data['standardized_syndrome'] = primary_syndrome
            standardized_patients.append(patient_data)

    df = pd.DataFrame(standardized_patients)
    print(f"生成 {len(df)} 例标准化患者数据")
    print(df[['patient_id', 'raw_note', 'standardized_syndrome', 'MMSE', 'pTau217']])

    # Step 4: TPS 特征工程（复用第一阶段代码）
    print("\n【步骤 3】多模态特征工程...")
    engineer = TPSFeatureEngineer()
    X, feature_names = engineer.fit_transform(df, TPSConfig())

    # Step 5: 训练/加载 TPS 模型
    print("\n【步骤 4】训练 TPS 评分模型...")
    y_dict = {syndrome: df[syndrome].values for syndrome in TPSConfig.SYNDROME_TYPES}
    tps_model = ExplainableTPSModel()
    tps_model.fit(X, y_dict, feature_names)

    # Step 6: 预测与解释
    print("\n【步骤 5】生成 TPS 评分与个性化解释...")
    predictions = tps_model.predict_score(X)

    for i in range(len(df)):
        patient_id = df.iloc[i]['patient_id']
        raw_note = df.iloc[i]['raw_note']
        std_syndrome = df.iloc[i]['standardized_syndrome']

        print(f"\n--- 患者 {patient_id} ---")
        print(f"原始描述: {raw_note}")
        print(f"标准化证候: {std_syndrome}")
        print(f"MMSE: {df.iloc[i]['MMSE']:.1f}, pTau217: {df.iloc[i]['pTau217']:.2f}")

        # TPS 评分
        scores = predictions.iloc[i]
        print("TPS 五维证候评分:")
        for syndrome in TPSConfig.SYNDROME_TYPES:
            marker = " <-- 标准化匹配" if syndrome == std_syndrome else ""
            print(f"  {syndrome}: {scores[syndrome]:.1f}/10{marker}")

        # 可解释性分析（针对标准化证候）
        explanation = tps_model.explain_instance(X[i], feature_names, std_syndrome)
        print(f"\n可解释性分析（{std_syndrome}）:")
        print(f"  关键驱动因素: {explanation['interpretation']}")

        # 推荐检测
        panel = std_engine.get_biomarker_panel([std_syndrome])
        print(f"  建议检测标志物: {', '.join(panel['recommended_tests']['upregulated_markers'][:3])}")

        # 生成可视化
        TPSVisualization.plot_syndrome_radar(scores, patient_id,
                                             save_path=f'tps_stage2_{patient_id}.png')

    print("\n" + "=" * 80)
    print("集成演示完成！已生成雷达图：tps_stage2_*.png")
    print("软著申请亮点：自由文本 → 标准证候 → AI 评分 → 检测推荐")
    print("=" * 80)


def generate_patient_by_syndrome(syndrome: str, confidence: float) -> dict:
    """
    根据证候类型生成模拟患者数据（体现不同证候的生物标志物差异）
    这是连接标准化模块和 TPS 模型的关键桥梁
    """
    np.random.seed(hash(syndrome) % 10000)

    # 基础数据模板
    data = {
        'age': np.random.normal(75, 5),
        'gender': np.random.choice(['M', 'F']),
        'disease_stage': np.random.choice(['MCI', 'mild_AD', 'moderate_AD'])
    }

    # 根据证候设置生物标志物特征（对应综述中的证候-肠脑轴关联）
    if syndrome == '肾虚髓减':
        data['MMSE'] = np.random.normal(18, 4)
        data['MoCA'] = np.random.normal(16, 3)
        data['pTau217'] = np.random.normal(2.5, 0.6)
        data['butyrate'] = np.random.normal(14, 4)  # 低
        data['IL1β'] = np.random.normal(3, 1)
        data['TMAO'] = np.random.normal(5, 1.5)

    elif syndrome == '痰瘀闭窍':
        data['MMSE'] = np.random.normal(16, 5)
        data['MoCA'] = np.random.normal(14, 4)
        data['pTau217'] = np.random.normal(2.2, 0.7)
        data['butyrate'] = np.random.normal(16, 5)
        data['IL1β'] = np.random.normal(5, 1.5)  # 高炎症
        data['TMAO'] = np.random.normal(7, 2)  # 高 TMAO
        data['zonulin'] = np.random.normal(20, 5)  # 屏障破坏

    elif syndrome == '心脾两虚':
        data['MMSE'] = np.random.normal(22, 3)
        data['MoCA'] = np.random.normal(20, 3)
        data['pTau217'] = np.random.normal(1.5, 0.5)
        data['butyrate'] = np.random.normal(18, 4)  # 相对高（营养问题）
        data['IL1β'] = np.random.normal(2, 0.8)
        data['sleep_quality'] = np.random.normal(4, 1)  # 睡眠差

    else:
        # 默认值
        data['MMSE'] = np.random.normal(20, 4)
        data['MoCA'] = np.random.normal(18, 3)
        data['pTau217'] = np.random.normal(2.0, 0.8)

    # 确保所有字段都有值
    for col in TPSConfig.COGNITIVE_TESTS + TPSConfig.GUT_BRAIN_MARKERS + TPSConfig.TCM_FEATURES:
        if col not in data:
            data[col] = np.random.normal(0, 1)

    # 证候评分（该证候高分，其他随机）
    for s in TPSConfig.SYNDROME_TYPES:
        if s == syndrome:
            data[s] = np.clip(np.random.normal(7.5, 1.5) * confidence, 0, 10)  # 高分
        else:
            data[s] = np.clip(np.random.normal(3, 1.5), 0, 10)  # 低分

    return data


if __name__ == "__main__":
    integrated_pipeline()
