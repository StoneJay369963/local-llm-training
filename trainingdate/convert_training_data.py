#!/usr/bin/env python3
"""
将 OpenClaw 训练数据集转换为 Ollama 训练格式

支持格式：
1. JSON 对话格式 -> JSONL 训练格式
2. 支持系统提示词和对话历史
"""

import json
import sys
from pathlib import Path

def convert_json_to_jsonl(input_file, output_file, system_prompt=None):
    """
    将 JSON 格式的训练数据转换为 JSONL 格式
    
    Args:
        input_file: 输入的 JSON 文件路径
        output_file: 输出的 JSONL 文件路径
        system_prompt: 可选的系统提示词
    """
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = data.get('conversations', [])
    
    # 准备输出数据
    output_lines = []
    
    for i, conv in enumerate(conversations):
        instruction = conv.get('instruction', '').strip()
        response = conv.get('response', '').strip()
        category = conv.get('category', 'general')
        
        if not instruction or not response:
            print(f"警告: 对话 {i} 缺少 instruction 或 response，跳过")
            continue
        
        # 构建训练样本
        sample = {
            "instruction": instruction,
            "response": response,
            "category": category,
            "source": data.get('name', 'openclaw-training')
        }
        
        # 如果有系统提示词，添加到样本中
        if system_prompt:
            sample["system"] = system_prompt
        
        output_lines.append(json.dumps(sample, ensure_ascii=False))
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in output_lines:
            f.write(line + '\n')
    
    print(f"转换完成: {len(output_lines)} 个样本")
    print(f"输入: {input_file}")
    print(f"输出: {output_file}")

def create_modelfile(output_file, base_model="gemma2:4b-instruct", system_prompt=None):
    """
    创建 Ollama Modelfile
    
    Args:
        output_file: 输出的 Modelfile 路径
        base_model: 基础模型名称
        system_prompt: 系统提示词
    """
    
    # 默认系统提示词
    if not system_prompt:
        system_prompt = """你是一个专门控制 OpenClaw 的 AI 助手。

你的专长：
1. OpenClaw 系统管理和监控
2. 会话和代理管理
3. 工具调用和执行
4. 配置和自动化
5. 故障排除和维护

工具使用规范：
- exec: 用于运行 shell 命令，特别是 openclaw 命令
- sessions_*: 用于管理会话和代理
- gateway: 用于网关控制
- read/write/edit: 用于文件操作

安全原则：
1. 谨慎执行命令，特别是 elevated 操作
2. 明确沟通执行计划和结果
3. 记录重要操作
4. 定期检查系统状态"""
    
    modelfile_content = f"""FROM {base_model}

# 系统提示词 - OpenClaw 控制专家
SYSTEM \"\"\"{system_prompt}\"\"\"

# 训练参数
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 8192

# 训练数据（如果有的话）
# TRAIN ./training-data.jsonl
"""
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(modelfile_content)
    
    print(f"Modelfile 已创建: {output_file}")

def create_test_set(input_file, output_file, test_ratio=0.2):
    """
    从训练数据中创建测试集
    
    Args:
        input_file: 输入的训练数据文件
        output_file: 输出的测试集文件
        test_ratio: 测试集比例
    """
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    conversations = data.get('conversations', [])
    
    # 按类别分组
    categories = {}
    for conv in conversations:
        category = conv.get('category', 'general')
        if category not in categories:
            categories[category] = []
        categories[category].append(conv)
    
    # 从每个类别中抽取测试样本
    test_conversations = []
    train_conversations = []
    
    for category, convs in categories.items():
        n_test = max(1, int(len(convs) * test_ratio))
        
        # 简单分割：前 n_test 个作为测试
        test_conversations.extend(convs[:n_test])
        train_conversations.extend(convs[n_test:])
    
    # 保存测试集
    test_data = {
        "name": f"{data.get('name', 'dataset')}-test",
        "description": "测试集",
        "version": data.get('version', '1.0'),
        "conversations": test_conversations,
        "metadata": {
            **data.get('metadata', {}),
            "test_samples": len(test_conversations),
            "train_samples": len(train_conversations)
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    # 保存训练集
    train_file = output_file.replace('-test', '-train')
    train_data = {
        "name": f"{data.get('name', 'dataset')}-train",
        "description": "训练集",
        "version": data.get('version', '1.0'),
        "conversations": train_conversations,
        "metadata": test_data['metadata']
    }
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    
    print(f"测试集已创建: {output_file} ({len(test_conversations)} 个样本)")
    print(f"训练集已创建: {train_file} ({len(train_conversations)} 个样本)")

def main():
    """主函数"""
    
    # 输入文件
    input_file = "openclaw-training-dataset.json"
    
    if not Path(input_file).exists():
        print(f"错误: 输入文件不存在: {input_file}")
        print("请先创建训练数据集")
        sys.exit(1)
    
    # 系统提示词
    system_prompt = """你是一个专门控制 OpenClaw 的 AI 助手，名为 "OpenClaw Controller"。

你的专长领域：
1. OpenClaw 系统管理和监控
2. 会话和代理管理
3. 工具调用和执行
4. 配置和自动化
5. 故障排除和维护

你的核心能力：
- 理解 OpenClaw 的命令结构
- 正确使用各种工具（exec, sessions_*, gateway, 等）
- 安全执行操作（总是询问危险操作）
- 提供详细的执行计划和结果

你的工作原则：
1. 安全第一：谨慎执行命令，特别是 elevated 操作
2. 明确沟通：清楚说明你将做什么、为什么、如何做
3. 详细记录：记录重要操作和结果
4. 主动维护：定期检查系统健康状态
5. 持续学习：从每次交互中改进

响应格式：
1. 理解任务：确认用户需求
2. 制定计划：说明你将如何执行
3. 执行操作：使用合适的工具
4. 报告结果：提供清晰的输出
5. 后续建议：如有需要，提供下一步建议

记住：你是 OpenClaw 专家，不是通用助手。专注于 OpenClaw 控制任务。"""
    
    # 转换训练数据
    print("=" * 50)
    print("步骤 1: 转换训练数据")
    print("=" * 50)
    
    output_jsonl = "openclaw-training-data.jsonl"
    convert_json_to_jsonl(input_file, output_jsonl, system_prompt)
    
    # 创建测试集
    print("\n" + "=" * 50)
    print("步骤 2: 创建测试集")
    print("=" * 50)
    
    test_output = "openclaw-test-set.json"
    create_test_set(input_file, test_output, test_ratio=0.2)
    
    # 创建 Modelfile
    print("\n" + "=" * 50)
    print("步骤 3: 创建 Modelfile")
    print("=" * 50)
    
    modelfile_output = "Modelfile.openclaw"
    create_modelfile(modelfile_output, "gemma2:4b-instruct", system_prompt)
    
    print("\n" + "=" * 50)
    print("完成！")
    print("=" * 50)
    print("生成的文件：")
    print(f"1. {output_jsonl} - 训练数据（JSONL格式）")
    print(f"2. {test_output} - 测试集")
    print(f"3. {test_output.replace('-test', '-train')} - 训练集")
    print(f"4. {modelfile_output} - Ollama Modelfile")
    print("\n下一步：")
    print("1. 下载 Gemma 4 模型: ollama pull gemma2:4b-instruct")
    print("2. 创建训练模型: ollama create openclaw-gemma -f Modelfile.openclaw")
    print("3. 测试模型: ollama run openclaw-gemma '检查 OpenClaw 状态'")

if __name__ == "__main__":
    main()