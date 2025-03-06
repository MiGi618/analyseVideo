import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import os
import librosa
import numpy as np
from huggingface_hub import HfFolder, try_to_load_from_cache
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME

# 全局变量
model = None
processor = None
pipe = None

def get_local_model_path(model_id, filename):
    """获取本地模型文件路径"""
    # 检查默认缓存目录
    cache_file = try_to_load_from_cache(model_id, filename)
    if cache_file and os.path.exists(cache_file):
        return os.path.dirname(cache_file)
    
    # 检查用户主目录下的缓存
    user_cache = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(user_cache):
        model_cache = os.path.join(user_cache, model_id)
        if os.path.exists(model_cache):
            return model_cache
            
    return None

def initialize_whisper():
    """初始化Whisper模型"""
    global model, processor, pipe
    
    if model is not None:
        return True  # 已经初始化过了
        
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    # 打印系统信息
    print("\n=== Whisper模型初始化 ===")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    model_id = "openai/whisper-large-v3"
    
    try:
        # 获取本地模型路径
        local_model_path = get_local_model_path(model_id, CONFIG_NAME)
        if not local_model_path:
            raise ValueError(f"未找到本地模型文件，请确保已下载模型: {model_id}")
            
        print(f"使用本地模型: {local_model_path}")
        
        # 记录模型加载时间
        start_time = time.time()
        
        # 从本地加载模型
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            local_model_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=False,
            local_files_only=True  # 强制使用本地文件
        ).to(device)
        
        processor = AutoProcessor.from_pretrained(
            local_model_path,
            local_files_only=True  # 强制使用本地文件
        )
        
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        load_time = time.time() - start_time
        print(f"模型加载完成，耗时: {load_time:.2f} 秒")
        
        return True
        
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        return False

def get_audio_info(file_path):
    """获取音频文件信息"""
    try:
        # 获取文件大小
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
        
        # 获取音频时长和采样率
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=audio_data, sr=sample_rate)
        
        # 计算音频统计信息
        mean_amplitude = np.mean(np.abs(audio_data))
        max_amplitude = np.max(np.abs(audio_data))
        
        print("\n音频文件信息:")
        print(f"文件大小: {file_size:.2f} MB")
        print(f"音频时长: {duration:.2f} 秒")
        print(f"采样率: {sample_rate} Hz")
        print(f"平均振幅: {mean_amplitude:.4f}")
        print(f"最大振幅: {max_amplitude:.4f}")
        
        return duration
        
    except Exception as e:
        print(f"获取音频信息时出错: {e}")
        return None

def save_transcription(text, output_dir="txt"):
    """保存转录文本到文件"""
    try:
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存文本
        output_path = os.path.join(output_dir, "output.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"\n转录文本已保存到: {output_path}")
        return output_path
    except Exception as e:
        print(f"保存转录文本时出错: {e}")
        return None

def process_audio(file_path):
    """处理音频文件并计时"""
    try:
        # 确保模型已初始化
        initialize_whisper()
        
        # 获取音频信息
        print(f"\n开始处理音频文件: {file_path}")
        duration = get_audio_info(file_path)
        
        # 记录转录开始时间
        transcribe_start = time.time()
        
        # 执行转录
        result = pipe(file_path)
        
        # 计算处理时间
        process_time = time.time() - transcribe_start
        
        print("\n处理结果:")
        print(f"转录用时: {process_time:.2f} 秒")
        if duration:
            print(f"实时率: {duration/process_time:.2f}x")
            print(f"每秒处理音频时长: {duration/process_time:.2f} 秒")
        
        # 保存转录文本
        if result and "text" in result:
            save_transcription(result["text"])
        
        return result
        
    except Exception as e:
        print(f"处理音频时出错: {e}")
        return None

if __name__ == "__main__":
    sample = r"F:\Whisper\audio\audio_59s.mp3"
    
    # 处理音频并获取结果
    result = process_audio(sample)
    
    if result:
        print("\n转录文本:")
        print(result["text"])
