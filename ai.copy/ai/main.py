from llm import *
from indextts.infer_v2 import IndexTTS2
import re
import subprocess

#uv run ai.copy\ai\main.py


tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints",
                use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)

def main():
    """主函数：命令行对话循环"""
    NO=0
    # 加载病人模板（使用第一个模板）
    patient_template = load_patient_template()

    # 构建系统提示
    system_prompt = build_system_prompt(patient_template)

    # 初始化对话历史
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]
    print_header()
    #视频序号/计数
    num_vidio=0
    #一整句话的数量
    num_he=0
    # 对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input("你: ").strip()

            # 检查退出命令
            if user_input.lower() in ['退出', 'exit', 'quit']:
                print("\n感谢使用，再见！")
                break

            # 检查清空命令
            if user_input.lower() in ['清空', 'clear']:
                # 重新加载病人模板
                patient_template = load_patient_template()
                system_prompt = build_system_prompt(patient_template)
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
                clear_screen()
                print_header()
                print("对话历史已清空！")
                continue

            # 跳过空输入
            if not user_input:
                continue

            # 添加用户消息到历史
            messages.append({
                "role": "user",
                "content": user_input
            })

            # 调用模型
            print("\nAI: ", end="", flush=True)
            ai_response= call_8b_model_openai(messages, stream=True)
            # 如果AI有响应，添加到历史
            if ai_response:
                messages.append({
                    "role": "assistant",
                    "content": ai_response
                })

                #初始化音频文件夹
                folder_path = "mp3\\"

                for filename in os.listdir(folder_path):
                    if filename.endswith(".wav"):
                        file_path = os.path.join(folder_path, filename)
                        os.remove(file_path)

                NO=0
                text = ai_response
                numbers, strings = process_string(text)
                for number in numbers:
                    tts.infer(spk_audio_prompt='examples/voice_10.wav', text=strings[NO],
                                   output_path="mp3\gen{}.wav".format(num_vidio), emo_vector=number, use_random=False,
                                   verbose=True)
                    num_vidio+=1
                    NO += 1


                # 将文件列表写入一个临时文本文件
                list_file_path = "mylist.txt"
                with open(list_file_path, "w", encoding="utf-8") as f:
                    for filename in os.listdir(folder_path):
                        f.write("file 'mp3\{}'".format(filename)+"\n")

                # 输出文件名
                output_file = "mp3_he\merged{}.wav".format(num_he)

                # 构建 FFmpeg 命令
                command = [
                    "ffmpeg",
                    "-f", "concat",  # 使用 concat 协议
                    "-safe", "0",  # 允许使用绝对路径（如果需要）
                    "-i", list_file_path,  # 指定列表文件为输入
                    "-c", "copy",  # 直接复制，不重新编码
                    "-y",  # 自动覆盖输出文件（可选）
                    output_file
                ]

                # 执行命令
                try:
                    subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"拼接成功，输出文件：{output_file}")
                    num_he+=1
                except subprocess.CalledProcessError as e:
                    print(f"拼接失败：{e.stderr}")

            print()  # 空行

        except KeyboardInterrupt:
            print("\n\n检测到中断，退出对话。")
            break
        except EOFError:
            print("\n\n输入结束，退出对话。")
            break
        except Exception as e:
            print(f"\n发生错误: {str(e)}")
            continue

def process_string(input_str):
    # 提取中括号内的数字
    pattern = r'\[(.*?)\]'
    brackets_content = re.findall(pattern, input_str)

    # 处理提取的内容，转换为数字列表
    numbers_list = []
    for content in brackets_content:
        # 分割字符串并转换为数字
        numbers = [float(num.strip()) for num in content.split(',')]
        numbers_list.append(numbers)

    # 删除中括号及其内容
    cleaned_str = re.sub(pattern, '', input_str)

    # 分割字符串为多个小字符串
    # 首先，使用中括号作为分隔符来获取分割点
    parts = re.split(pattern, input_str)
    # 过滤掉空字符串，只保留中括号外的文本
    string_parts = [part for part in parts if part and not re.match(r'^\d+(\.\d+)?(,\d+(\.\d+)?)*$', part)]

    return numbers_list, string_parts

if __name__ == "__main__":
    main()


