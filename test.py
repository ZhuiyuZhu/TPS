import json
import pandas as pd
from collections import Counter

# 加载全部数据
files = ['Train_TCM_Data_v1.json', 'Validation_TCM_Data_v1.json', 'Test_TCM_Data_v1.json']
all_records = []
for f in files:
    with open(f, 'r', encoding='utf-8') as fp:
        all_records.extend(json.load(fp))

print(f"总病例数: {len(all_records)}")

# 提取所有证候（注意：证候是多标签，用分号分隔）
all_syndromes = []
ad_related_records = []

ad_keywords = ['痴呆', '呆病', '健忘', '阿尔茨海默', 'AD', '髓减', '脑髓', '神呆']

for record in all_records:
    syndrome_str = record.get('TCM Syndrome', '')
    clinical = record.get('Clinical Information', '') + record.get('Clinical Data', '')

    # 分割多标签证候
    syndromes = [s.strip() for s in syndrome_str.split(';') if s.strip()]
    all_syndromes.extend(syndromes)

    # 检查是否 AD 相关
    is_ad = any(kw in syndrome_str or kw in clinical for kw in ad_keywords)
    if is_ad:
        ad_related_records.append({
            'id': record.get('Medical Record ID'),
            'syndrome': syndrome_str,
            'clinical': clinical[:100] + '...'
        })

# 统计证候分布
syndrome_counts = Counter(all_syndromes)
print(f"\n共有 {len(syndrome_counts)} 种不同证候，Top 20：")
for syndrome, count in syndrome_counts.most_common(20):
    print(f"  {syndrome}: {count}例")

# 查看是否有 AD 相关病例
print(f"\nAD相关病例数: {len(ad_related_records)}")
if ad_related_records:
    for case in ad_related_records[:5]:
        print(f"  - {case['id']}: {case['syndrome']}")
        print(f"    摘要: {case['clinical']}")
else:
    print("  未找到明确标注为痴呆/AD的病例（这很正常，该数据集是通用中医病例）")
