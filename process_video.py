import os
import time
from datetime import datetime
from getAudio import extract_audio
from hugWhisper import process_audio
from getConclusion import process_transcription, initialize_client
import subprocess

def create_output_dirs():
    """创建所需的输出目录"""
    dirs = ['video', 'audio', 'txt', 'notes', 'stats', 'conversations']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"已创建或确认目录存在: {dir_name}")

def process_video(video_path, api_key=None, base_url=None, model_name="deepseek-r1-250120"):
    """
    处理视频的主流程函数
    
    主要步骤：
    1. 从视频中提取音频 (getAudio.py)
    2. 将音频转换为文本 (hugWhisper.py)
    3. 使用AI分析文本内容 (getConclusion.py)
    
    参数:
        video_path (str): 输入视频文件的路径
        api_key (str): 火山大模型API密钥
        base_url (str): 火山大模型Base URL
        model_name (str): 使用的模型名称
    
    返回:
        dict: 包含处理结果的字典
    """
    try:
        # 验证视频文件路径
        if not os.path.exists(video_path):
            raise Exception(f"视频文件不存在: {video_path}")
        
        print(f"\n开始处理视频文件: {video_path}")
        print(f"文件大小: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
        
        # 记录开始时间
        total_start_time = time.time()
        
        # 创建输出目录
        create_output_dirs()
        
        # 步骤1：提取音频 (getAudio.py -> extract_audio)
        print("\n=== 步骤1：提取音频 ===")
        audio_path = os.path.join("audio", f"audio_{int(time.time())}.mp3")
        print(f"正在从视频中提取音频...")
        print(f"输出路径: {audio_path}")
        
        audio_result = extract_audio(video_path, audio_path)
        if not audio_result:
            raise Exception("音频提取失败，请检查视频文件是否完整或是否已安装FFmpeg")
        print(f"音频提取完成: {audio_result}")
        
        # 步骤2：语音识别 (hugWhisper.py -> process_audio)
        print("\n=== 步骤2：语音识别 ===")
        print(f"正在使用Whisper模型转录音频...")
        transcription_result = process_audio(audio_result)
        
        if not transcription_result:
            raise Exception("语音识别失败，请检查音频文件是否正常")
        if "text" not in transcription_result:
            raise Exception("语音识别结果格式错误")
            
        txt_file = os.path.join("txt", "output.txt")
        if not os.path.exists(txt_file):
            raise Exception("转录文本文件未生成")
        print(f"语音识别完成，文本已保存到: {txt_file}")
        
        # 步骤3：内容分析 (getConclusion.py -> process_transcription)
        print("\n=== 步骤3：内容分析 ===")
        print("正在初始化AI模型...")
        
        # 设置API配置
        if not api_key:
            api_key = os.getenv("ARK_API_KEY")
            if not api_key:
                raise Exception("未提供API密钥，且环境变量ARK_API_KEY未设置")
        
        if not base_url:
            base_url = "https://ark.cn-beijing.volces.com/api/v3/"
            
        # 初始化大模型客户端
        client = initialize_client(api_key, base_url)
        if not client:
            raise Exception("AI模型初始化失败，请检查API密钥和Base URL是否正确")
            
        print("AI模型初始化完成")
        
        print("开始分析文本内容...")
        analysis_result = process_transcription(txt_file, model_name)
        if not analysis_result:
            raise Exception("内容分析失败，请检查API配置和文本内容")
        print(f"内容分析完成，报告已保存到: {analysis_result}")
        
        # 计算总处理时间
        total_time = time.time() - total_start_time
        
        # 准备处理结果
        result = {
            "status": "success",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_path": video_path,
            "audio_path": audio_result,
            "transcription_path": txt_file,
            "analysis_path": analysis_result,
            "total_time": total_time
        }
        
        print("\n=== 处理完成 ===")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"处理结果:")
        print(f"- 视频文件: {os.path.basename(video_path)}")
        print(f"- 音频文件: {os.path.basename(audio_result)}")
        print(f"- 转录文本: {os.path.basename(txt_file)}")
        print(f"- 分析报告: {os.path.basename(analysis_result)}")
        
        return result
        
    except Exception as e:
        print(f"\n处理过程中出错: {str(e)}")
        return {
            "status": "error",
            "error_message": str(e),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

def main():
    """主函数"""
    try:
        print("\n=== AI视频分析系统 ===")
        print("本系统将自动完成以下步骤：")
        print("1. 从视频中提取音频 (需要安装FFmpeg)")
        print("2. 使用Whisper模型进行语音识别")
        print("3. 使用大模型进行内容分析和总结")
        
        # 检查FFmpeg是否已安装
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("\n错误：未检测到FFmpeg！")
            print("请先安装FFmpeg并将其添加到系统环境变量中")
            print("下载地址：https://ffmpeg.org/download.html")
            return
            
        print("\n=== 环境检查 ===")
        print("√ FFmpeg 已安装")
        
        print("\n请按照提示输入必要信息：")
        
        # 1. 获取视频文件路径
        while True:
            video_path = input("\n请输入视频文件路径（输入q退出）: ").strip()
            if video_path.lower() == 'q':
                print("程序已退出")
                return
            
            if not os.path.exists(video_path):
                print("错误：文件不存在，请重新输入")
                continue
                
            if not video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
                print("警告：不支持的视频格式，支持的格式：mp4, avi, mov, mkv, flv")
                if input("是否继续？(y/n): ").lower() != 'y':
                    continue
            break
        
        # 2. 获取API配置
        print("\n=== API配置 ===")
        api_key = os.getenv("ARK_API_KEY")
        if api_key:
            print("检测到环境变量ARK_API_KEY")
            use_env = input("是否使用环境变量中的API密钥？(y/n): ").lower() == 'y'
            if not use_env:
                api_key = input("请输入火山大模型API密钥: ").strip()
        else:
            print("未检测到环境变量ARK_API_KEY")
            api_key = input("请输入火山大模型API密钥: ").strip()
            if not api_key:
                print("错误：未提供API密钥")
                return
        
        base_url = input("\n请输入火山大模型Base URL（直接回车使用默认值）: ").strip() or None
        
        # 3. 确认信息
        print("\n=== 处理信息确认 ===")
        print(f"视频文件: {video_path}")
        print(f"文件大小: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
        print(f"API密钥: {'环境变量' if use_env else '手动输入'}")
        print(f"Base URL: {base_url or '默认值'}")
        
        if input("\n确认开始处理？(y/n): ").lower() != 'y':
            print("已取消处理")
            return
        
        # 4. 处理视频
        result = process_video(video_path, api_key, base_url)
        
        # 5. 输出处理状态
        if result["status"] == "success":
            print("\n=== 视频处理成功完成！===")
            print(f"总耗时: {result['total_time']:.2f} 秒")
            print("\n生成的文件：")
            print(f"- 音频文件: {os.path.basename(result['audio_path'])}")
            print(f"- 转录文本: {os.path.basename(result['transcription_path'])}")
            print(f"- 分析报告: {os.path.basename(result['analysis_path'])}")
            
            # 询问是否打开生成的文件
            print("\n是否打开生成的文件？")
            if input("1. 打开分析报告 (y/n): ").lower() == 'y':
                os.system(f"start {result['analysis_path']}")
            if input("2. 打开转录文本 (y/n): ").lower() == 'y':
                os.system(f"start {result['transcription_path']}")
        else:
            print(f"\n处理失败: {result['error_message']}")
            print("请检查错误信息并重试")
            
    except KeyboardInterrupt:
        print("\n\n程序已被用户中断")
    except Exception as e:
        print(f"\n程序运行出错: {str(e)}")
    finally:
        print("\n程序结束")

if __name__ == "__main__":
    main() 