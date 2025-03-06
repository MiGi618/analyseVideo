import subprocess
import os


## 提取音频
## 输入: 视频文件路径
## 输出: 音频文件路径   
## 用法:
## result = extract_audio(r"F:\Whisper\video\testVideo_59s.mp4", r"F:\Whisper\audio\output_audio.mp3")
## 如果成功, result 为音频文件路径, 否则为 None
def extract_audio(input_path: str, output_path: str = None) -> str:
    """
    使用 FFmpeg 从视频文件中提取 MP3 格式的音频
    
    参数:
        input_path (str): 输入视频文件的路径
        output_path (str, 可选): 输出音频文件的路径，默认与输入文件同目录
    
    返回:
        str: 成功时返回输出文件路径，失败时返回 None
    
    异常:
        会触发常规异常并打印错误信息
    """
    try:
        # 检查 FFmpeg 是否可用
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError("未找到 FFmpeg 或版本不兼容，请先安装 FFmpeg 并添加到系统路径")

    # 验证输入文件是否存在
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    # 设置默认输出路径
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.mp3"
    else:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    # 构建 FFmpeg 命令
    command = [
        "ffmpeg",
        "-y",  # 覆盖输出文件不提示
        "-i", input_path,
        "-vn",  # 不处理视频流
        "-codec:a", "libmp3lame",  # 使用 LAME MP3 编码器
        "-q:a", "0",  # 最高音频质量 (VBR 0-9, 0=best)
        "-map_metadata", "0",  # 保留元数据
        output_path
    ]

    try:
        # 执行转换命令
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"音频提取成功: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg 错误 ({e.returncode}):\n{e.stderr}"
    except Exception as e:
        error_msg = f"意外错误: {str(e)}"

    print(f"提取失败: {error_msg}")
    return None

# # 使用示例
# if __name__ == "__main__":
#     result = extract_audio(r"F:\Whisper\video\testVideo_59s.mp4", r"F:\Whisper\audio\output_audio.mp3")
#     if result:
#         print(f"生成文件: {result}")