import random

from UltraEval.tasks.postprocess import GSM8KPost


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = f"[INST] <<SYS>>\nคุณเป็นผู้ช่วยอัจฉริยะด้านภาษาไทย-หน้าที่ของคุณคือช่วยเหลือผู้ใช้ในการแก้ไขปัญหาของพวกเขา-ในการตอบคำถาม-คุณจะต้องใช้ภาษาไทยในการตอบเพื่อให้ผู้ใช้ชาวไทยเข้าใจคำตอบของคุณได้ดีขึ้น-คำตอบควรมีประโยชน์-อ่านง่าย-และไม่ควรมีเนื้อหาที่เป็นอันตราย-โปรดทราบว่าคุณไม่ควรใช้ภาษาอังกฤษในการตอบคำถาม\n<</SYS>>\n\n{data['question']} [/INST]"
    correct_answer = data["answer"]
    gsm8kp = GSM8KPost()
    _, processed_correct_answer = gsm8kp([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
