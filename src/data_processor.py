"""
数据处理器模块
用于文本数据的清洗、分词和格式化
"""

import re
import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """数据处理配置"""
    min_text_length: int = 100
    max_text_length: int = 100000
    remove_html: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    normalize_whitespace: bool = True
    lowercase: bool = False
    remove_special_chars: bool = False
    min_sentence_count: int = 3


class TextCleaner:
    """文本清洗器"""

    def __init__(self, config: ProcessingConfig):
        self.config = config

        # HTML标签正则
        self.html_pattern = re.compile(
            r'<[^>]+>|&[a-zA-Z]+;|&#\d+;',
            re.IGNORECASE
        )

        # URL正则
        self.url_pattern = re.compile(
            r'https?://\S+|www\.\S+',
            re.IGNORECASE
        )

        # 邮箱正则
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )

        # 多余空白字符
        self.whitespace_pattern = re.compile(r'\s+')

        # 控制字符
        self.control_pattern = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]')

    def clean(self, text: str) -> str:
        """执行所有清洗步骤"""
        if not text:
            return ""

        # 移除HTML标签
        if self.config.remove_html:
            text = self.html_pattern.sub(' ', text)

        # 移除URL
        if self.config.remove_urls:
            text = self.url_pattern.sub(' ', text)

        # 移除邮箱
        if self.config.remove_emails:
            text = self.email_pattern.sub(' ', text)

        # 移除控制字符
        text = self.control_pattern.sub('', text)

        # 标准化空白字符
        if self.config.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text)

        # 转小写
        if self.config.lowercase:
            text = text.lower()

        # 移除特殊字符
        if self.config.remove_special_chars:
            text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:\'"()\-]', ' ', text)

        # 去除首尾空白
        text = text.strip()

        return text

    def validate(self, text: str) -> bool:
        """验证文本是否有效"""
        if not text:
            return False

        # 检查长度
        if len(text) < self.config.min_text_length:
            return False
        if len(text) > self.config.max_text_length:
            return False

        # 检查句子数量
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) < self.config.min_sentence_count:
            return False

        return True


def process_single_file(
    file_path: str,
    output_path: str,
    config: ProcessingConfig
) -> Dict[str, int]:
    """处理单个文件"""
    cleaner = TextCleaner(config)
    stats = {"total": 0, "valid": 0, "invalid": 0, "lines": 0}

    with open(file_path, 'r', encoding='utf-8') as infile:
        with open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                stats["lines"] += 1
                text = line.strip()

                if not text:
                    stats["invalid"] += 1
                    continue

                stats["total"] += 1
                cleaned = cleaner.clean(text)

                if cleaner.validate(cleaned):
                    outfile.write(cleaned + '\n')
                    stats["valid"] += 1
                else:
                    stats["invalid"] += 1

    return stats


def deduplicate_texts(texts: List[str], threshold: float = 0.8) -> List[str]:
    """
    基于MinHash的文本去重
    threshold: 相似度阈值，超过此值认为重复
    """
    try:
        from datasketch import MinHash, MinHashLSH

        # 创建MinHash索引
        lsh = MinHashLSH(threshold=threshold, num_perm=128)
        minhashes = {}

        for i, text in enumerate(texts):
            tokens = set(text.split())
            m = MinHash(num_perm=128)
            for token in tokens:
                m.update(token.encode('utf-8'))

            if f"doc_{i}" not in lsh:
                lsh.insert(f"doc_{i}", m)
                minhashes[f"doc_{i}"] = m

        # 收集去重后的文本
        seen = set()
        unique_texts = []

        for i, text in enumerate(texts):
            doc_id = f"doc_{i}"
            if doc_id not in seen:
                unique_texts.append(text)
                seen.add(doc_id)

                # 标记相似的文档
                for j in lsh.query(minhashes.get(doc_id, MinHash())):
                    seen.add(j)

        return unique_texts

    except ImportError:
        logger.warning("datasketch未安装，跳过去重步骤")
        return texts


def split_dataset(
    data_path: str,
    train_ratio: float = 0.9,
    output_dir: str = "./data"
) -> tuple:
    """划分训练集和验证集"""
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    import random
    random.shuffle(lines)

    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    eval_lines = lines[split_idx:]

    train_path = os.path.join(output_dir, "train.txt")
    eval_path = os.path.join(output_dir, "eval.txt")

    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    with open(eval_path, 'w', encoding='utf-8') as f:
        f.writelines(eval_lines)

    logger.info(f"训练集: {len(train_lines)} 条, 验证集: {len(eval_lines)} 条")
    return train_path, eval_path


def convert_to_chat_format(
    texts: List[str],
    format_type: str = "chatml"
) -> List[Dict]:
    """
    转换为对话格式
    format_type: chatml, alpaca, sharegpt
    """
    if format_type == "chatml":
        return [
            {
                "messages": [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": ""}
                ]
            }
            for text in texts
        ]
    elif format_type == "alpaca":
        return [
            {
                "instruction": "请处理以下文本",
                "input": "",
                "output": text
            }
            for text in texts
        ]
    else:
        raise ValueError(f"不支持的格式: {format_type}")


def download_public_dataset(
    dataset_name: str,
    output_dir: str = "./data"
) -> str:
    """下载公开数据集"""
    os.makedirs(output_dir, exist_ok=True)

    if dataset_name == "tiny_shakespere":
        # 小型Shakespeare数据集
        try:
            import torchtext
            from torchtext.datasets import WikiText2
            train_iter = WikiText2(split='train')
            output_path = os.path.join(output_dir, "wiki_train.txt")

            with open(output_path, 'w', encoding='utf-8') as f:
                for line in train_iter:
                    if line.strip():
                        f.write(line)

            logger.info(f"已下载WikiText2数据集到 {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"下载失败: {e}")
            raise

    elif dataset_name == "alpaca":
        # Alpaca数据集
        try:
            from datasets import load_dataset
            dataset = load_dataset("yahma/alpaca-cleaned", split="train")
            output_path = os.path.join(output_dir, "alpaca_train.json")

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([{"text": item["text"]} for item in dataset], f, ensure_ascii=False)

            logger.info(f"已下载Alpaca数据集到 {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"下载失败: {e}")
            raise

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def stream_text_file(file_path: str, chunk_size: int = 1000) -> Iterator[List[str]]:
    """流式读取大文件"""
    chunk = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunk.append(line)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


class DatasetBuilder:
    """数据集构建器"""

    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.cleaner = TextCleaner(config)

    def build_from_directory(
        self,
        input_dir: str,
        output_file: str,
        extensions: List[str] = ['.txt', '.md', '.json']
    ) -> Dict[str, int]:
        """从目录构建数据集"""
        stats = {"files": 0, "lines": 0, "valid": 0}

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for ext in extensions:
                for file_path in Path(input_dir).rglob(f'*{ext}'):
                    stats["files"] += 1

                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            for line in infile:
                                stats["lines"] += 1
                                text = line.strip()

                                if not text:
                                    continue

                                cleaned = self.cleaner.clean(text)
                                if self.cleaner.validate(cleaned):
                                    outfile.write(cleaned + '\n')
                                    stats["valid"] += 1

                    except Exception as e:
                        logger.warning(f"处理文件失败 {file_path}: {e}")
                        continue

        logger.info(f"构建完成: {stats}")
        return stats

    def build_conversation_dataset(
        self,
        qa_pairs: List[tuple],
        output_file: str,
        format_type: str = "chatml"
    ) -> None:
        """构建对话数据集"""
        dataset = []

        for question, answer in qa_pairs:
            if format_type == "chatml":
                item = {
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": answer}
                    ]
                }
            elif format_type == "alpaca":
                item = {
                    "instruction": question,
                    "input": "",
                    "output": answer
                }
            else:
                item = {"text": f"Q: {question}\nA: {answer}"}

            dataset.append(item)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        logger.info(f"对话数据集已保存: {output_file}, 共 {len(dataset)} 条")


if __name__ == "__main__":
    # 示例用法
    config = ProcessingConfig(
        min_text_length=100,
        max_text_length=50000,
        remove_html=True,
        remove_urls=True,
        normalize_whitespace=True
    )

    # 构建数据集
    builder = DatasetBuilder(config)

    # 从公开数据集下载示例数据
    try:
        dataset_path = download_public_dataset("tiny_shakespere")
        print(f"数据集路径: {dataset_path}")
    except Exception as e:
        print(f"下载示例数据失败，请手动准备数据: {e}")
