from openai import OpenAI
import os
import json
import time
from datetime import datetime
import tiktoken

def initialize_client(api_key, base_url):
    """初始化API客户端"""
    global client
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        # 测试API连接
        response = client.chat.completions.create(
            model="deepseek-r1-250120",
            messages=[{"role": "user", "content": "测试连接"}],
            max_tokens=10
        )
        print("API连接测试成功")
        return client
    except Exception as e:
        print(f"API初始化失败: {str(e)}")
        return None

# 全局客户端变量
client = None

def count_tokens(text, model="deepseek-r1-250120"):
    """计算文本的 token 数量"""
    # 注意：这里可能需要根据实际模型调整
    encoding = tiktoken.encoding_for_model("gpt-4")  # 使用兼容的编码器
    return len(encoding.encode(text))

def split_text(text, max_tokens=4000):
    """将文本分段，确保每段不超过最大 token 限制"""
    # 按句号分割文本
    sentences = text.split("。")
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip() + "。"
        sentence_tokens = count_tokens(sentence)
        
        if current_length + sentence_tokens > max_tokens:
            # 当前块已满，保存并开始新块
            chunks.append("".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_tokens
        else:
            # 添加句子到当前块
            current_chunk.append(sentence)
            current_length += sentence_tokens
    
    # 添加最后一个块
    if current_chunk:
        chunks.append("".join(current_chunk))
    
    return chunks

def save_statistics(stats, output_dir="stats"):
    """保存统计信息到TXT文件"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"stats_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== 处理统计信息 ===\n\n")
            f.write(f"处理时间: {stats['timestamp']}\n")
            f.write("\n=== 文件信息 ===\n")
            f.write(f"文件名: {stats['file_info']['name']}\n")
            f.write(f"文件大小: {stats['file_info']['size']} 字符\n")
            f.write(f"分段数量: {stats['file_info']['chunks']} 段\n")
            
            f.write("\n=== 处理时间统计 ===\n")
            f.write(f"总耗时: {stats['timing']['total']:.2f} 秒\n")
            f.write(f"读取文件: {stats['timing']['read']:.2f} 秒\n")
            f.write(f"文本分段: {stats['timing']['split']:.2f} 秒\n")
            f.write(f"生成思维导图: {stats['timing']['mindmap']:.2f} 秒\n")
            f.write(f"生成文本分析: {stats['timing']['analysis']:.2f} 秒\n")
            
            f.write("\n=== Token统计 ===\n")
            f.write("思维导图:\n")
            f.write(f"  - 输入: {stats['tokens']['mindmap']['input']} tokens\n")
            f.write(f"  - 输出: {stats['tokens']['mindmap']['output']} tokens\n")
            f.write("文本分析:\n")
            f.write(f"  - 输入: {stats['tokens']['analysis']['input']} tokens\n")
            f.write(f"  - 输出: {stats['tokens']['analysis']['output']} tokens\n")
            f.write("总计:\n")
            f.write(f"  - 输入: {stats['tokens']['total']['input']} tokens\n")
            f.write(f"  - 输出: {stats['tokens']['total']['output']} tokens\n")
        
        print(f"统计信息已保存到: {filepath}")
        return filepath
    except Exception as e:
        print(f"保存统计信息时出错: {e}")
        return None

def save_conversation_history(conversations, output_dir="conversations"):
    """保存对话记录到TXT文件"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("=== 对话记录 ===\n\n")
            f.write(f"时间: {conversations['timestamp']}\n")
            f.write(f"文件: {conversations['file']}\n")
            f.write(f"模型: {conversations['model']}\n\n")
            
            # 写入思维导图对话
            f.write("=== 思维导图生成 ===\n\n")
            for conv in conversations['mindmap_conversations']:
                if conv['type'] == 'mindmap':
                    f.write(f"--- 第 {conv['part']} 段文本处理 ---\n")
                else:
                    f.write("--- 合并处理 ---\n")
                
                f.write("\n系统提示:\n")
                f.write(conv['messages'][0]['content'] + "\n")
                
                f.write("\n用户输入:\n")
                f.write(conv['messages'][1]['content'] + "\n")
                
                f.write("\n模型回答:\n")
                f.write(conv['response'] + "\n")
                
                f.write(f"\nToken统计:\n")
                f.write(f"- 输入: {conv['input_tokens']} tokens\n")
                f.write(f"- 输出: {conv['output_tokens']} tokens\n")
                f.write("\n" + "="*50 + "\n\n")
            
            # 写入文本分析对话
            f.write("=== 文本分析生成 ===\n\n")
            for conv in conversations['analysis_conversations']:
                if conv['type'] == 'analysis':
                    f.write(f"--- 第 {conv['part']} 段文本分析 ---\n")
                else:
                    f.write("--- 合并分析 ---\n")
                
                f.write("\n系统提示:\n")
                f.write(conv['messages'][0]['content'] + "\n")
                
                f.write("\n用户输入:\n")
                f.write(conv['messages'][1]['content'] + "\n")
                
                f.write("\n模型回答:\n")
                f.write(conv['response'] + "\n")
                
                f.write(f"\nToken统计:\n")
                f.write(f"- 输入: {conv['input_tokens']} tokens\n")
                f.write(f"- 输出: {conv['output_tokens']} tokens\n")
                f.write("\n" + "="*50 + "\n\n")
        
        print(f"对话记录已保存到: {filepath}")
        return filepath
    except Exception as e:
        print(f"保存对话记录时出错: {e}")
        return None

def create_markdown_mindmap(text_chunks, model_name="deepseek-r1-250120"):
    """使用火山大模型分段生成思维导图（多轮对话形式）"""
    try:
        all_mindmaps = []
        conversations = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        # 初始化对话历史
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的内容分析师，请将给定的文本整理成markdown格式的可预览的思维导图。可以使用mermaid"
            }
        ]
        
        # 首先处理每个文本块
        for i, chunk in enumerate(text_chunks, 1):
            print(f"正在处理第 {i}/{len(text_chunks)} 段文本...")
            
            # 添加用户输入到对话历史
            messages.append({
                "role": "user",
                "content": f"请将以下文本整理成思维导图格式（这是文本的第{i}部分，共{len(text_chunks)}部分）：\n\n{chunk}"
            })
            
            # 记录对话
            current_conversation = {
                "type": "mindmap",
                "part": i,
                "messages": messages.copy()  # 复制当前的对话历史
            }
            
            # 计算输入tokens
            input_tokens = sum(count_tokens(msg["content"]) for msg in messages)
            total_input_tokens += input_tokens
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            
            # 将助手的回答添加到对话历史
            messages.append({
                "role": "assistant",
                "content": content
            })
            
            # 计算输出tokens
            output_tokens = count_tokens(content)
            total_output_tokens += output_tokens
            
            all_mindmaps.append(content)
            
            # 记录响应
            current_conversation["response"] = content
            current_conversation["input_tokens"] = input_tokens
            current_conversation["output_tokens"] = output_tokens
            conversations.append(current_conversation)
            
            time.sleep(1)  # 避免触发 API 限制
        
        # 然后生成一个总结性的思维导图
        if len(all_mindmaps) > 1:
            combined_mindmap = "\n\n".join(all_mindmaps)
            
            # 添加用户请求合并的消息
            messages.append({
                "role": "user",
                "content": f"请将以上所有思维导图整合成一个完整的、层次清晰的思维导图。保持相同的格式，但要去除重复的内容，使其更加连贯。\n\n{combined_mindmap}"
            })
            
            # 记录合并对话
            current_conversation = {
                "type": "mindmap_merge",
                "messages": messages.copy()
            }
            
            # 计算输入tokens
            input_tokens = sum(count_tokens(msg["content"]) for msg in messages)
            total_input_tokens += input_tokens
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            
            # 计算输出tokens
            output_tokens = count_tokens(content)
            total_output_tokens += output_tokens
            
            # 记录响应
            current_conversation["response"] = content
            current_conversation["input_tokens"] = input_tokens
            current_conversation["output_tokens"] = output_tokens
            conversations.append(current_conversation)
            
            final_content = content
        else:
            final_content = all_mindmaps[0]
        
        return final_content, conversations, total_input_tokens, total_output_tokens
    except Exception as e:
        print(f"生成思维导图时出错: {e}")
        return None, [], 0, 0

def create_text_analysis(text_chunks, model_name="deepseek-r1-250120"):
    """使用火山大模型分段生成文本分析（多轮对话形式）"""
    try:
        all_analyses = []
        conversations = []
        total_input_tokens = 0
        total_output_tokens = 0
        
        # 初始化对话历史
        messages = [
            {
                "role": "system",
                "content": "你是一个专业的内容分析师，请对给定的文本进行深入分析，包括：主要内容、关键观点、逻辑分析和重要信息。"
            }
        ]
        
        # 分析每个文本块
        for i, chunk in enumerate(text_chunks, 1):
            print(f"正在分析第 {i}/{len(text_chunks)} 段文本...")
            
            # 添加用户输入到对话历史
            messages.append({
                "role": "user",
                "content": f"请分析以下文本（这是文本的第{i}部分，共{len(text_chunks)}部分）：\n\n{chunk}"
            })
            
            # 记录对话
            current_conversation = {
                "type": "analysis",
                "part": i,
                "messages": messages.copy()
            }
            
            # 计算输入tokens
            input_tokens = sum(count_tokens(msg["content"]) for msg in messages)
            total_input_tokens += input_tokens
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            
            # 将助手的回答添加到对话历史
            messages.append({
                "role": "assistant",
                "content": content
            })
            
            # 计算输出tokens
            output_tokens = count_tokens(content)
            total_output_tokens += output_tokens
            
            all_analyses.append(content)
            
            # 记录响应
            current_conversation["response"] = content
            current_conversation["input_tokens"] = input_tokens
            current_conversation["output_tokens"] = output_tokens
            conversations.append(current_conversation)
            
            time.sleep(1)
        
        # 生成总体分析
        if len(all_analyses) > 1:
            combined_analysis = "\n\n".join(all_analyses)
            
            # 添加用户请求合并的消息
            messages.append({
                "role": "user",
                "content": f"请根据以上所有分析结果，生成一个完整的总体分析。需要整合所有重要观点，去除重复内容，使分析更加连贯和全面。\n\n{combined_analysis}"
            })
            
            # 记录合并对话
            current_conversation = {
                "type": "analysis_merge",
                "messages": messages.copy()
            }
            
            # 计算输入tokens
            input_tokens = sum(count_tokens(msg["content"]) for msg in messages)
            total_input_tokens += input_tokens
            
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.7,
                max_tokens=2000,
                stream=True
            )
            
            content = ""
            for chunk in response:
                if chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            
            # 计算输出tokens
            output_tokens = count_tokens(content)
            total_output_tokens += output_tokens
            
            # 记录响应
            current_conversation["response"] = content
            current_conversation["input_tokens"] = input_tokens
            current_conversation["output_tokens"] = output_tokens
            conversations.append(current_conversation)
            
            final_content = content
        else:
            final_content = all_analyses[0]
        
        return final_content, conversations, total_input_tokens, total_output_tokens
    except Exception as e:
        print(f"生成文本分析时出错: {e}")
        return None, [], 0, 0

def save_to_markdown(mindmap, analysis, text, output_dir="notes"):
    """保存结果到 Markdown 文件"""
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"note_{timestamp}.md"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("# 内容分析报告\n\n")
            f.write("## 原文内容\n\n")
            f.write(f"```\n{text}\n```\n\n")
            f.write("## 思维导图\n\n")
            f.write(f"{mindmap}\n\n")
            f.write("## 内容分析\n\n")
            f.write(f"{analysis}\n")
            
            # 添加元信息
            f.write("\n---\n")
            f.write(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"文本长度：{len(text)} 字符\n")
            f.write(f"Token 数量：{count_tokens(text)} tokens\n")
            
            # 添加处理时间信息到文件
            if 'total_time' in locals():
                f.write(f"\n## 处理时间统计\n")
                f.write(f"- 总耗时：{total_time:.2f}秒\n")
                f.write(f"- 读取文件：{read_time:.2f}秒\n")
                f.write(f"- 文本分段：{split_time:.2f}秒\n")
                f.write(f"- 生成思维导图：{mindmap_time:.2f}秒\n")
                f.write(f"- 生成文本分析：{analysis_time:.2f}秒\n")
                f.write(f"- 保存文件：{save_time:.2f}秒\n")
        
        print(f"笔记已保存到: {filepath}")
        return filepath
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return None

def process_transcription(text_file, model_name="deepseek-r1-250120"):
    """处理转录文本文件"""
    try:
        total_start_time = time.time()
        
        # 读取文本文件
        read_start_time = time.time()
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()
        read_time = time.time() - read_start_time
        print(f"读取文件耗时: {read_time:.2f}秒")
        
        # 分割文本
        split_start_time = time.time()
        print("正在分析文本长度并进行分段...")
        text_chunks = split_text(text)
        split_time = time.time() - split_start_time
        print(f"文本已分为 {len(text_chunks)} 段，分段耗时: {split_time:.2f}秒")
        
        # 生成思维导图
        mindmap_start_time = time.time()
        print("正在生成思维导图...")
        mindmap, mindmap_conversations, mindmap_input_tokens, mindmap_output_tokens = create_markdown_mindmap(text_chunks, model_name)
        mindmap_time = time.time() - mindmap_start_time
        print(f"生成思维导图耗时: {mindmap_time:.2f}秒")
        
        # 生成文本分析
        analysis_start_time = time.time()
        print("正在生成文本分析...")
        analysis, analysis_conversations, analysis_input_tokens, analysis_output_tokens = create_text_analysis(text_chunks, model_name)
        analysis_time = time.time() - analysis_start_time
        print(f"生成文本分析耗时: {analysis_time:.2f}秒")
        
        if mindmap and analysis:
            # 保存结果
            save_start_time = time.time()
            
            # 准备统计信息
            total_time = time.time() - total_start_time
            stats = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file_info": {
                    "name": os.path.basename(text_file),
                    "size": len(text),
                    "chunks": len(text_chunks)
                },
                "timing": {
                    "total": total_time,
                    "read": read_time,
                    "split": split_time,
                    "mindmap": mindmap_time,
                    "analysis": analysis_time
                },
                "tokens": {
                    "mindmap": {
                        "input": mindmap_input_tokens,
                        "output": mindmap_output_tokens
                    },
                    "analysis": {
                        "input": analysis_input_tokens,
                        "output": analysis_output_tokens
                    },
                    "total": {
                        "input": mindmap_input_tokens + analysis_input_tokens,
                        "output": mindmap_output_tokens + analysis_output_tokens
                    }
                }
            }
            
            # 保存统计信息
            stats_file = save_statistics(stats)
            
            # 保存对话记录
            conversations = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "file": os.path.basename(text_file),
                "model": model_name,
                "mindmap_conversations": mindmap_conversations,
                "analysis_conversations": analysis_conversations
            }
            conversation_file = save_conversation_history(conversations)
            
            # 保存Markdown文件
            filepath = save_to_markdown(mindmap, analysis, text)
            save_time = time.time() - save_start_time
            
            if filepath:
                print("\n处理完成！")
                print(f"总耗时: {total_time:.2f}秒")
                print(f"详细耗时统计:")
                print(f"- 读取文件: {read_time:.2f}秒")
                print(f"- 文本分段: {split_time:.2f}秒")
                print(f"- 生成思维导图: {mindmap_time:.2f}秒")
                print(f"- 生成文本分析: {analysis_time:.2f}秒")
                print(f"- 保存文件: {save_time:.2f}秒")
                print(f"\nToken统计:")
                print(f"- 思维导图: 输入 {mindmap_input_tokens} / 输出 {mindmap_output_tokens}")
                print(f"- 文本分析: 输入 {analysis_input_tokens} / 输出 {analysis_output_tokens}")
                print(f"- 总计: 输入 {mindmap_input_tokens + analysis_input_tokens} / 输出 {mindmap_output_tokens + analysis_output_tokens}")
                return filepath
    except Exception as e:
        print(f"处理文本时出错: {e}")
        return None

def main(api_key=None, base_url=None, model_name="deepseek-r1-250120"):
    """主函数"""
    global client
    
    # 如果没有提供API密钥，则请求输入
    if not api_key:
        api_key = input("请输入你的火山大模型 API 密钥: ").strip()
    if not base_url:
        base_url = input("请输入火山大模型的 Base URL: ").strip()
    
    # 初始化客户端
    client = initialize_client(api_key, base_url)
    
    # 获取转录文本文件路径
    text_file = input("请输入转录文本文件路径: ").strip()
    
    if not os.path.exists(text_file):
        print("文件不存在！")
        return
    
    # 处理文本
    result_file = process_transcription(text_file, model_name)
    
    if result_file:
        print(f"\n处理结果已保存到: {result_file}")
        # 自动打开生成的文件
        os.system(f"start {result_file}")

# if __name__ == "__main__":
#     main(os.getenv("ARK_API_KEY"), "https://ark.cn-beijing.volces.com/api/v3/")