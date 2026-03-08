# from llm import *
# from indextts.infer_v2 import IndexTTS2
#
# def main():
#     """主函数：命令行对话循环"""
#
#     # 加载病人模板（使用第一个模板）
#     patient_template = load_patient_template()
#
#     # 构建系统提示
#     system_prompt = build_system_prompt(patient_template)
#
#     # 初始化对话历史
#     messages = [
#         {
#             "role": "system",
#             "content": system_prompt
#         }
#     ]
#
#     print_header()
#
#     # 对话循环
#     while True:
#         try:
#             # 获取用户输入
#             user_input = input("你: ").strip()
#
#             # 检查退出命令
#             if user_input.lower() in ['退出', 'exit', 'quit']:
#                 print("\n感谢使用，再见！")
#                 break
#
#             # 检查清空命令
#             if user_input.lower() in ['清空', 'clear']:
#                 # 重新加载病人模板
#                 patient_template = load_patient_template()
#                 system_prompt = build_system_prompt(patient_template)
#                 messages = [
#                     {
#                         "role": "system",
#                         "content": system_prompt
#                     }
#                 ]
#                 clear_screen()
#                 print_header()
#                 print("对话历史已清空！")
#                 continue
#
#             # 跳过空输入
#             if not user_input:
#                 continue
#
#             # 添加用户消息到历史
#             messages.append({
#                 "role": "user",
#                 "content": user_input
#             })
#
#             # 调用模型
#             print("\nAI: ", end="", flush=True)
#             ai_response= call_8b_model_openai(messages, stream=True)
#             # 如果AI有响应，添加到历史
#             if ai_response:
#                 messages.append({
#                     "role": "assistant",
#                     "content": ai_response
#                 })
#             print()  # 空行
#
#         except KeyboardInterrupt:
#             print("\n\n检测到中断，退出对话。")
#             break
#         except EOFError:
#             print("\n\n输入结束，退出对话。")
#             break
#         except Exception as e:
#             print(f"\n发生错误: {str(e)}")
#             continue
#
# if __name__ == "__main__":
#     main()
#
from indextts.infer_v2 import IndexTTS2
tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)
"""
[愤怒:0.6]你们是怎么看病的？！
"""

text = "哇塞！这个爆率也太高了！!!!!欧皇附体了!!!!"
tts.infer(spk_audio_prompt='examples/voice_10.wav', text=text, output_path="mp3\gen1.wav", emo_vector=[0, 0, 0, 0, 0, 0, 0.45, 0], use_random=False, verbose=True)


