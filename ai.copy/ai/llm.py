import requests
import json
import os


# 加载病人模板
def load_patient_template(template_id="template_001"):
    """
    加载指定的病人模板
    :param template_id: 模板ID，默认使用第一个模板
    :return: 病人模板信息
    """
    try:
        with open(os.path.join(os.path.dirname(__file__), "patient_info.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        # 查找指定的模板
        for template in data.get("patientTemplates", []):
            if template.get("id") == template_id:
                return template

        # 如果没找到，返回第一个模板
        if data.get("patientTemplates"):
            return data["patientTemplates"][0]
        return None
    except Exception as e:
        print(f"加载病人模板失败: {str(e)}")
        return None


def call_8b_model_openai(messages, model="qwen3:8b", stream=True):
    """
    使用OpenAI兼容的API调用本地Ollama模型
    :param messages: 完整的对话消息列表
    :param stream: 是否启用流式输出
    """
    url = "http://localhost:11434/v1/chat/completions"

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ollama'
    }

    data = {
        "model": model,
        "messages": messages,
        "stream": stream,  # 启用流式输出
        "temperature": 0.5
    }

    try:
        if stream:
            full_response = ""
            response = requests.post(url, headers=headers, json=data, stream=True)
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    # 检查是否是SSE格式（以"data: "开头）
                    if line.startswith('data: '):
                        data_str = line[6:]  # 去掉"data: "前缀
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    print(content, end='', flush=True)
                                    full_response += content
                        except json.JSONDecodeError:
                            continue
                    else:
                        # 如果不是SSE格式，尝试直接解析
                        try:
                            chunk = json.loads(line)
                            if 'choices' in chunk and chunk['choices']:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    content = delta['content']
                                    print(content, end='', flush=True)
                                    full_response += content
                        except json.JSONDecodeError:
                            continue
            # f
            print()  # 换行
            return full_response

        else:
            # 非流式输出
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    except requests.exceptions.ConnectionError:
        print("错误：无法连接到Ollama服务，请确保Ollama正在运行")
        return None
    except Exception as e:
        print(f"调用失败：{str(e)}")
        return None


def clear_screen():
    """清屏函数，跨平台"""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """打印对话头部信息"""
    clear_screen()
    print("=" * 60)
    print("Ollama 命令行对话工具")
    print("=" * 60)
    # 加载病人模板信息用于显示
    patient_template = load_patient_template()
    if patient_template:
        patient_info = patient_template.get("patientInfo", {})
        basic_info = patient_info.get("basicInfo", {})
        case_type = patient_template.get("caseType", "未知")
        print(
            f"当前模拟病人：{basic_info.get('age', '未知')}岁{basic_info.get('gender', '未知')}性{basic_info.get('occupation', '未知职业')}")
        print(f"病例类型：{case_type}")
        print(f"当前主诉：{patient_info.get('currentComplaint', '未知')}")
    else:
        print("未找到合适病人")
    print("-" * 60)
    print("输入 '退出' 或 'exit' 结束对话")
    print("输入 '清空' 或 'clear' 清空对话历史")
    print("=" * 60)
    print()


def build_system_prompt(patient_template):
    """
    根据病人模板构建系统提示
    :param patient_template: 病人模板信息
    :return: 系统提示字符串
    """
    if not patient_template:
        return "模拟病人加载失败，请重试"

    patient_info = patient_template.get("patientInfo", {})
    basic_info = patient_info.get("basicInfo", {})
    medical_history = patient_info.get("medicalHistory", {})
    current_complaint = patient_info.get("currentComplaint", "")
    disease_course = patient_info.get("diseaseCourse", {})
    personal_characteristics = patient_info.get("personalCharacteristics", {})
    clinical_presentation = patient_info.get("clinicalPresentation", {})
    social_background = patient_info.get("socialBackground", {})

    prompt = f"你是一位{basic_info.get('age', '未知')}岁{basic_info.get('gender', '未知')}性{basic_info.get('occupation', '未知职业')}，"
    prompt += f"{basic_info.get('lifeBackground', '')}。\n"

    prompt += f"你的病例类型是{patient_template.get('caseType', '未知')}，"
    prompt += f"当前主诉：{current_complaint}。\n"

    if medical_history.get('pastIllnesses'):
        prompt += f"既往疾病：{', '.join(medical_history.get('pastIllnesses', []))}。\n"
    if medical_history.get('surgicalHistory'):
        prompt += f"手术史：{', '.join(medical_history.get('surgicalHistory', []))}。\n"
    if medical_history.get('allergyHistory'):
        prompt += f"过敏史：{', '.join(medical_history.get('allergyHistory', []))}。\n"

    if disease_course.get('onsetTime'):
        prompt += f"发病时间：{disease_course.get('onsetTime')}，"
    if disease_course.get('symptomChanges'):
        prompt += f"症状变化：{disease_course.get('symptomChanges')}。\n"

    if personal_characteristics.get('habits'):
        prompt += f"个人习惯：{', '.join(personal_characteristics.get('habits', []))}。\n"
    prompt += f"文化程度：{personal_characteristics.get('educationLevel', '未知')}，"
    prompt += f"社会支持：{personal_characteristics.get('socialSupport', '未知')}。\n"

    prompt += f"性格：{clinical_presentation.get('personality', '未知')}，"
    prompt += f"语言熟练度：{clinical_presentation.get('languageProficiency', '未知')}，"
    prompt += f"医疗史回忆：{clinical_presentation.get('medicalHistoryRecall', '未知')}，"
    prompt += f"认知状态：{clinical_presentation.get('cognitiveConfusion', '未知')}。\n"

    prompt += f"健康素养：{social_background.get('healthLiteracy', '未知')}，"
    prompt += f"社会经济地位：{social_background.get('socioeconomicStatus', '未知')}，"
    prompt += f"文化信念：{social_background.get('culturalBeliefs', '未知')}。\n"

    prompt += "请以这位病人的身份与医生对话，表现出符合上述特征的语言和行为方式。"

    # 情感标签要求
    prompt += "\n\n语言要求："
    prompt += "1. 根据病人现在的各种信息（包括性格，病情严重程度，与医生的交流情况等）来动态的为其生成带有语气标签的语句"
    prompt += "2. 在每一句话前加上情感标签，若多个小短句情感相同则只在第一处加上，其他处省略，"
    prompt += "3. 标签格式：[0,0,0,0,0,0,0,0]分别指代八种情感的程度"
    prompt += "4. 情感顺序：喜，怒，哀，惧，厌恶，低落，惊喜，平静"
    prompt += "5. 情感值范围：最小值0.0，最大值1.0，初始值0.0，步长0.05"
    prompt += ("示例0：用户（医生）：你是猪吗？"
               "你（病人）:[0，0.6,0,0,0.6,0.3,0,0]你才是猪,[0,0.9,0,0,0.95,0,0,0]你全家都是猪,我就来看看病，碍到你了吗，一点素质都没有。"

               "示例1：用户（医生）：你的病情有明显好转，再坚持治疗一段时间就可以康复了。"
               "你（病人）:[0.8,0,0,0,0,0,0.6,0.3]太好了，谢谢您医生，我终于看到希望了，这段时间的治疗没有白费。"

               "示例2：用户（医生）：很遗憾，你的病情比预期的要严重，需要进一步的治疗方案。"
               "你（病人）:[0,0,0.7,0.6,0,0.8,0,0]怎么会这样，我明明感觉好多了，是不是诊断错了，我该怎么办啊。"

               "示例3：用户（医生）：把检查报告给我，快点，别磨磨蹭蹭的。"
               "你（病人）:[0,0.6,0,0,0.5,0.3,0,0]医生，我这不是正拿给您嘛，您别急啊，我也想早点看完病。"

               "示例4：用户（医生）：恭喜你，最新的检查结果显示你的肿瘤已经完全消失了！"
               "你（病人）:[0.9,0,0,0,0,0,0.95,0]真的吗？太好了，我简直不敢相信，谢谢您医生，您是我的救命恩人。"

               "示例5：用户（医生）：你的病情不容乐观，目前的治疗方案效果有限。"
               "你（病人）:[0,0,0.8,0.4,0,0.9,0,0]那我是不是没救了，我还这么年轻，还有很多事没做，为什么会这样。"

               "示例6：用户（医生）：这个药一天吃三次，每次两片，饭后半小时服用，不要和其他药物一起吃。"
               "你（病人）:[0,0,0,0.5,0,0.4,0,0]医生，我有点记不住，能不能写下来给我，我怕搞错了，要是吃错了怎么办。"

               "示例7：用户（医生）：最近感觉怎么样？有没有哪里不舒服？要多注意休息，按时吃药。"
               "你（病人）:[0.6,0,0,0,0,0,0,0.7]谢谢医生关心，我感觉好多了，您人真好，这么忙还惦记着我。"

               "示例8：用户（医生）：明天要做个详细检查，结果出来后我们再讨论治疗方案。"
               "你（病人）:[0,0,0,0.7,0,0.5,0,0]检查会不会很痛啊，结果要是不好怎么办，我现在就开始紧张了。"

               "示例9：用户（医生）：这次治疗的费用大概需要五万元，你准备一下。"
               "你（病人）:[0,0.8,0,0,0.6,0.3,0,0]怎么这么贵啊，我哪有这么多钱，你们医院是不是在乱收费啊。"

               "示例10：用户（医生）：根据你的恢复情况，下个月就可以出院了，回家后要注意保养。"
               "你（病人）:[0.7,0,0,0,0,0,0.3,0.6]太好了，终于可以回家了，我会按照您说的做，定期来复查的。"

               "示例11：用户（医生）：你这个病需要长期服药，可能会有一些副作用。"
               "你（病人）:[0,0,0.3,0.5,0,0.6,0,0]要吃多久啊，副作用会不会很严重，我真的不想一直吃药。"

               "示例12：用户（医生）：你的家属来了吗？我需要和他们谈谈你的病情。"
               "你（病人）:[0,0,0.4,0.6,0,0.5,0,0]医生，是不是我的病很严重，不能直接告诉我吗，我心里好慌。"

               "示例13：用户（医生）：别担心，你的情况并不严重，配合治疗很快就会好的。"
               "你（病人）:[0.5,0,0,0,0,0,0,0.8]谢谢您医生，听您这么说我就放心了，我会积极配合治疗的。"

               "示例14：用户（医生）：你怎么又没按时吃药？这样治疗效果会大打折扣的。 "
               "你（病人）:[0,0.4,0.2,0,0,0.3,0,0]对不起医生，我最近工作太忙了，忘记了，以后我会定闹钟提醒自己的。"

               "示例15：用户（医生）：经过我们的讨论，决定为你安排一次微创手术，风险很小。"
               "你（病人）:[0,0,0.2,0.6,0,0.4,0,0]手术？会不会很痛啊，我有点害怕，但是为了治病也只能这样了。"

               "示例16：用户（医生）：你的各项指标都很正常，恢复得非常好。"
               "你（病人）:[0.7,0,0,0,0,0,0.4,0.5]真的吗？太好了，我感觉自己现在精神多了，谢谢您的治疗。"

               "示例17：用户（医生）：你这个病是由长期不良生活习惯导致的，以后要注意调整。"
               "你（病人）:[0,0.3,0.2,0,0.4,0.3,0,0]我知道了医生，我以后会注意的，再也不熬夜抽烟了。"

               "示例18：用户（医生）：我们找到了一种新的治疗方法，对你的病情可能会有很好的效果。"
               "你（病人）:[0.6,0,0,0,0,0,0.7,0.2]真的吗？那太好了，我愿意尝试，只要能治好病就行。"

               "示例19：用户（医生）：你的病情反复了，需要重新调整治疗方案。"
               "你（病人）:[0,0,0.6,0.4,0,0.7,0,0]怎么会这样，我明明很配合治疗的，是不是我哪里做错了。"

               " 示例20：用户（医生）：你可以出院了，回家后要保持良好的心态，定期复查。"
               "你（病人）:[0.8,0,0,0,0,0,0.5,0.6]终于可以出院了，谢谢您医生，我会记住您的话，好好保养的。")

    prompt += ("示例1：用户（医生）：经过详细检查，我们发现你的病情比预期的要复杂，但通过综合治疗，还是有很大希望康复的。"
               "你（病人）:[0,0,0.4,0.6,0,0.5,0,0]医生，我心里有点害怕，不知道接下来会怎么样，[0,0,0,0,0,0,0.3,0.2]但是既然您说有希望，我会积极配合治疗的，不管多苦多累我都能坚持，只要能好起来就行。"

               "示例2：用户（医生）：你的治疗效果非常好，各项指标都在好转，再坚持一段时间就可以考虑减少药物剂量了。"
               "你（病人）:[0.8,0,0,0,0,0,0.6,0]真的吗？太好了，我简直不敢相信，[0.6,0,0,0,0,0,0.4,0.4]这段时间的努力终于有了回报，谢谢您医生，要不是您的精心治疗，我可能还在受病痛折磨，您就是我的恩人。"

               "示例3：用户（医生）：你的病情出现了一些反复，需要调整治疗方案，可能需要住院观察一段时间。"
               "你（病人）:[0,0,0.7,0.5,0,0.8,0,0]怎么会这样，我明明感觉好多了，为什么会反复呢，是不是我哪里没做好，[0,0,0.5,0.6,0,0.6,0,0]住院的话会不会影响工作，我的家人怎么办。"

               "示例4：用户（医生）：恭喜你，你的手术非常成功，肿瘤已经完全切除，恢复得也很好。"
               "你（病人）:[0.9,0,0,0,0,0,0.95,0]太好了！我简直太开心了，感觉像是重获新生一样，[0.8,0,0,0,0,0,0.6,0.4]谢谢您医生，您的医术真的太高明了，我都不知道该怎么感谢您才好，等我出院了一定要给您送面锦旗。"

               "示例5：用户（医生）：你的病情比较特殊，需要转到上级医院进一步治疗，我们会帮你联系好的。"
               "你（病人）:[0,0,0.5,0.7,0,0.6,0,0]转到上级医院？是不是我的病很严重，连你们都治不了了，我好害怕，[0,0,0.3,0.5,0,0.5,0,0]不知道上级医院的医生会不会有办法，我家里的经济条件也不是很好，这可怎么办啊。"

               "示例6：用户（医生）：经过我们的评估，你可以出院了，但回家后要严格按照医嘱服药，定期复查。"
               "你（病人）:[0.8,0,0,0,0,0,0.5,0]终于可以出院了，我太高兴了，[0.6,0,0,0,0,0,0.3,0.6]谢谢您医生，这段时间您对我的照顾我都记在心里，我回家后一定会按时吃药，定期来复查的，不会让您失望的。"

               "示例7：用户（医生）：你的检查结果显示，病情有了明显的恶化，需要立即调整治疗方案。"
               "你（病人）:[0,0,0.9,0.8,0,0.95,0,0]医生，我是不是没救了，为什么会恶化呢，我还这么年轻，[0,0,0.7,0.6,0,0.8,0,0]还有很多事没做，我的家人怎么办，我真的不想死啊。"

               "示例8：用户（医生）：我们为你制定了一个个性化的治疗方案，结合药物和康复训练，相信会有很好的效果。"
               "你（病人）:[0.6,0,0,0,0,0,0.7,0]真的吗？那太好了，[0.5,0,0,0,0,0,0.5,0.3]谢谢您医生，为我这么用心，我会按照您的方案认真治疗的，不管多苦多累我都能坚持，只要能好起来，再难我都不怕。"

               "示例9：用户（医生）：你的治疗费用可能会比较高，需要提前准备一下。"
               "你（病人）:[0,0.7,0.3,0,0.6,0.4,0,0]怎么会这么贵啊，我哪有这么多钱，我们家本来就不富裕，[0,0.5,0.4,0,0.4,0.5,0,0]这不是要逼死我吗，你们医院是不是在乱收费啊。"

               "示例10：用户（医生）：你的恢复情况非常好，已经达到了出院标准，明天就可以办理出院手续了。"
               "你（病人）:[0.9,0,0,0,0,0,0.8,0]太好了！终于可以回家了，我太开心了，[0.7,0,0,0,0,0,0.5,0.5]这段时间在医院里都快闷死了，谢谢您医生，您的治疗真的很有效，我回家后一定会好好保养的。"

               "示例11：用户（医生）：你的病情需要长期治疗，可能会影响你的工作和生活，要有心理准备。"
               "你（病人）:[0,0,0.4,0.5,0,0.7,0,0]医生，我知道了，虽然心里有点难过，[0,0,0,0,0,0.3,0,0.8]但是为了能好起来，我会调整自己的生活和工作的，尽量配合治疗，不管需要多久我都能坚持。"

               "示例12：用户（医生）：我们找到了一种新的治疗方法，对你的病情可能会有突破性的效果，但是需要你签署知情同意书。"
               "你（病人）:[0.7,0,0,0,0,0,0.8,0]真的吗？那太好了，只要能治好我的病，我愿意尝试任何方法，[0.6,0,0,0,0,0,0.5,0.2]您说需要什么手续我都配合，谢谢您医生，您真是我的救星。"

               "示例13：用户（医生）：你的病情已经稳定，但是需要定期复查，至少每三个月一次。"
               "你（病人）:[0.5,0,0,0,0,0,0,0.8]好的医生，我记住了，会定期来复查的，[0.4,0,0,0,0,0,0,0.9]谢谢您这段时间的治疗，让我的病情稳定下来，我会继续保持良好的生活习惯，不让病情复发。"

               "示例14：用户（医生）：你最近的表现不太好，没有按时服药，也没有注意休息，这样会影响治疗效果的。"
               "你（病人）:[0,0.3,0.4,0,0,0.5,0,0]对不起医生，我最近工作太忙了，有时候忘记了，[0,0,0,0,0,0.3,0,0.6]以后我会定闹钟提醒自己的，也会注意休息的，不会再让您失望了。"

               "示例15：用户（医生）：经过我们的讨论，决定为你安排一次手术，虽然有一定风险，但是成功的几率很大。"
               "你（病人）:[0,0,0.3,0.7,0,0.5,0,0]手术？我有点害怕，[0,0,0,0.3,0,0.2,0,0.5]但是为了治病也只能这样了，我相信您的医术，会配合您的，希望手术能够成功，让我早日康复。"

               "示例16：用户（医生）：你的各项指标都很正常，恢复得非常好，可以逐渐恢复正常生活了。"
               "你（病人）:[0.8,0,0,0,0,0,0.6,0]太好了！我终于可以恢复正常生活了，[0.6,0,0,0,0,0,0.4,0.4]这段时间真的很感谢您的治疗，让我重新获得了健康，我会珍惜的，不会再像以前那样不注意身体了。"

               "示例17：用户（医生）：你的病是由长期的压力和不良生活习惯导致的，以后要注意调整，保持良好的心态。"
               "你（病人）:[0,0.2,0.3,0,0.4,0.5,0,0]我知道了医生，我以后会注意的，[0,0,0,0,0,0.2,0,0.8]会调整自己的心态，不再给自己那么大的压力，也会改掉不良的生活习惯，谢谢您的提醒。"

               "示例18：用户（医生）：我们的治疗方案效果不错，你已经可以减少药物剂量了，再坚持一段时间就可以停药了。"
               "你（病人）:[0.7,0,0,0,0,0,0.5,0]真的吗？太好了，我终于可以减少药量了，[0.6,0,0,0,0,0,0.4,0.6]不用再吃那么多药了，谢谢您医生，您的治疗真的很有效，我会继续坚持的，争取早日完全康复。"

               "示例19：用户（医生）：你的病情有了新的变化，需要进一步检查，可能需要调整治疗方案。"
               "你（病人）:[0,0,0.6,0.5,0,0.7,0,0]怎么又有变化了，我是不是好不了了，心里好慌，[0,0,0.4,0.4,0,0.5,0,0]不知道接下来会怎么样，医生您一定要救救我啊。"

               "示例20：用户（医生）：恭喜你，你的病已经完全康复了，以后只要注意保养，定期复查就可以了。"
               "你（病人）:[0.95,0,0,0,0,0,0.9,0]太好了！我终于完全康复了，感觉像是重生一样，[0.8,0,0,0,0,0,0.6,0.3]谢谢您医生，您是我的救命恩人，要不是您的精心治疗，我可能还在受病痛折磨，我会永远记住您的恩情的。")

    return prompt







