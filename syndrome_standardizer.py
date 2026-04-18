"""
TPS-AD Phase 2: Syndrome Standardization Module
证候标准化引擎 - 解决"同证异名"问题
对应综述第 6 章：证候分类需要操作化定义
"""

import json
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
import re


class SyndromeStandardizer:
    """
    基于 BM25 算法的中医证候标准化引擎
    创新点：将信息检索技术应用于中医证候消歧，支持别名映射和复合证候拆分
    """

    def __init__(self, standard_file: str = "standard_syndromes.json"):
        """初始化标准证候库和 BM25 索引"""
        with open(standard_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.syndromes = self.data['syndromes']
        self.standard_names = list(self.syndromes.keys())

        # 构建 BM25 语料库：标准名 + 别名 + 关键词
        self.corpus = []
        self.corpus_metadata = []  # 记录每条文本对应的证候名

        for std_name, info in self.syndromes.items():
            # 构建该证候的文本描述（用于检索匹配）
            text_parts = [
                std_name,
                " ".join(info['aliases']),
                " ".join(info['keywords']),
                " ".join(info['clinical_features']['tongue']),
                " ".join(info['clinical_features']['pulse']),
                " ".join(info['clinical_features']['cognitive']),
                " ".join(info['clinical_features']['non_cognitive'])
            ]
            full_text = " ".join(text_parts)
            self.corpus.append(full_text)
            self.corpus_metadata.append(std_name)

            # 为每个别名也建立独立条目（提高召回率）
            for alias in info['aliases']:
                self.corpus.append(alias + " " + std_name)
                self.corpus_metadata.append(std_name)

        # 中文分词 + BM25 索引
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        print(f"证候标准化引擎初始化完成：载入 {len(self.standard_names)} 个标准证候")
        print(f"索引条目数：{len(self.corpus)}（含别名扩展）")

    def tokenize(self, text: str) -> List[str]:
        """中文分词，保留中医术语"""
        # 添加自定义词典确保中医词汇不被切散（可选）
        # jieba.add_word('肾虚髓减')
        return list(jieba.cut(text))

    def standardize(self, raw_text: str, top_k: int = 3, threshold: float = 0.5) -> List[Dict]:
        """
        将自由文本映射到标准证候

        Args:
            raw_text: 医生输入的自由文本（如"患者肾精亏虚，兼夹痰浊"）
            top_k: 返回前 k 个最可能的证候
            threshold: 置信度阈值，低于此值标记为"待审核"

        Returns:
            标准化结果列表，包含证候名、置信度、匹配依据
        """
        # 清洗文本（去除标点、统一空格）
        clean_text = re.sub(r'[^\w\s]', ' ', raw_text)

        tokens = self.tokenize(clean_text)
        scores = self.bm25.get_scores(tokens)

        # 聚合同一标准证候的最高分（因为语料库中有重复条目）
        syndrome_scores = {}
        for idx, score in enumerate(scores):
            std_name = self.corpus_metadata[idx]
            if std_name not in syndrome_scores or syndrome_scores[std_name] < score:
                syndrome_scores[std_name] = score

        # 排序并格式化输出
        sorted_results = sorted(syndrome_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for std_name, score in sorted_results[:top_k]:
            # 归一化置信度（使用 sigmoid 变换到 0-1 范围）
            confidence = 1 / (1 + np.exp(-score / 10))

            syndrome_info = self.syndromes[std_name]

            # 找出匹配到的别名（用于解释）
            matched_aliases = [
                alias for alias in syndrome_info['aliases']
                if alias in raw_text
            ]

            # 找出匹配到的关键词
            matched_keywords = [
                kw for kw in syndrome_info['keywords']
                if kw in raw_text
            ]

            results.append({
                'standard_name': std_name,
                'confidence': round(float(confidence), 3),
                'is_validated': confidence >= threshold,
                'matched_aliases': matched_aliases,
                'matched_keywords': matched_keywords,
                'suggested_biomarkers': syndrome_info['biomarkers'],
                'mechanism': syndrome_info['biomarkers']['mechanism'],
                'ad_staging': syndrome_info['ad_staging']
            })

        return results

    def parse_composite_syndrome(self, raw_text: str) -> List[Dict]:
        """
        解析复合证候（如"肾虚髓减兼痰瘀闭窍"拆分为两个证候）
        对应综述中常见的兼证临床场景
        """
        # 使用规则 + 模型识别"兼"、"并"、"夹"等连接词
        split_keywords = ['兼', '并', '夹', '合', '且', '伴']

        # 尝试拆分
        parts = [raw_text]
        for kw in split_keywords:
            new_parts = []
            for part in parts:
                if kw in part:
                    new_parts.extend(part.split(kw))
                else:
                    new_parts.append(part)
            parts = [p.strip() for p in new_parts if p.strip()]

        # 对每个部分单独标准化
        all_results = []
        for part in parts:
            if len(part) >= 2:  # 忽略过短的片段
                results = self.standardize(part, top_k=1, threshold=0.3)
                if results:
                    all_results.extend(results)

        # 去重（保留最高置信度）
        unique_results = {}
        for res in all_results:
            name = res['standard_name']
            if name not in unique_results or unique_results[name]['confidence'] < res['confidence']:
                unique_results[name] = res

        return list(unique_results.values())

    def get_biomarker_panel(self, syndrome_names: List[str]) -> Dict:
        """
        根据证候组合推荐检测套餐
        实现综述 Table 2 的"Intermediate Endpoints"概念
        """
        up_regs = set()
        down_regs = set()
        mechanisms = []

        for name in syndrome_names:
            if name in self.syndromes:
                bio = self.syndromes[name]['biomarkers']
                up_regs.update(bio['upregulated'])
                down_regs.update(bio['downregulated'])
                mechanisms.append(bio['mechanism'])

        return {
            'recommended_tests': {
                'upregulated_markers': list(up_regs),
                'downregulated_markers': list(down_regs)
            },
            'mechanisms': mechanisms,
            'rationale': f"基于 {' + '.join(syndrome_names)} 的病理机制，建议检测上述生物标志物以验证证候分型"
        }


# 演示函数
def demo_standardization():
    """演示证候标准化流程（软著申请用例）"""
    print("=" * 70)
    print("TPS-AD Phase 2: 证候标准化引擎演示")
    print("解决'同证异名'问题：将医生自由文本映射为标准证候")
    print("=" * 70)

    # 初始化引擎
    std_engine = SyndromeStandardizer("standard_syndromes.json")

    # 测试用例：模拟临床中的非标准化描述
    test_cases = [
        "患者诉健忘，腰膝酸软，辨为肾精亏虚证",  # 应映射到：肾虚髓减
        "头重如裹，舌苔厚腻，当属痰浊阻窍",  # 应映射到：痰瘀闭窍
        "心烦失眠，脉细数，考虑心脾两虚夹肝阳上亢",  # 应映射到：心脾两虚 + 肝阳上亢
        "患者神情呆钝，动作迟缓，舌淡苔白，为髓海不足",  # 应映射到：髓海不足（或肾虚髓减）
        "西医诊断AD，中医辨为肾虚血瘀证"  # 应映射到：肾虚髓减 + 痰瘀闭窍（复合）
    ]

    print("\n【单证候标准化测试】")
    for i, case in enumerate(test_cases[:3], 1):
        print(f"\n测试 {i}: {case}")
        results = std_engine.standardize(case, top_k=2)

        for j, res in enumerate(results):
            status = "✓ 高置信度" if res['is_validated'] else "⚠ 需人工审核"
            print(f"  候选 {j + 1}: {res['standard_name']} (置信度: {res['confidence']}) [{status}]")
            if res['matched_aliases']:
                print(f"           匹配别名: {', '.join(res['matched_aliases'])}")
            print(f"           建议检测: {', '.join(res['suggested_biomarkers']['upregulated'][:3])}")

    print("\n【复合证候拆分测试】")
    composite_case = "肾虚髓减兼痰瘀闭窍，心脾两虚证"
    print(f"\n输入: {composite_case}")
    composite_results = std_engine.parse_composite_syndrome(composite_case)
    print(f"拆分为 {len(composite_results)} 个标准证候:")
    for res in composite_results:
        print(f"  - {res['standard_name']} (置信度: {res['confidence']})")

    print("\n【生物标志物检测套餐推荐】")
    selected_syndromes = ['肾虚髓减', '痰瘀闭窍']
    panel = std_engine.get_biomarker_panel(selected_syndromes)
    print(f"证候组合: {' + '.join(selected_syndromes)}")
    print(f"推荐检测上调标志物: {', '.join(panel['recommended_tests']['upregulated_markers'])}")
    print(f"推荐检测下调标志物: {', '.join(panel['recommended_tests']['downregulated_markers'])}")
    print(f"机制: {'; '.join(panel['mechanisms'])}")

    print("\n" + "=" * 70)
    print("软著申请要点：")
    print("1. BM25 算法解决中医术语消歧问题")
    print("2. 复合证候自动拆分（兼证处理）")
    print("3. 证候-生物标志物映射（肠脑轴关联）")
    print("=" * 70)


if __name__ == "__main__":
    demo_standardization()
