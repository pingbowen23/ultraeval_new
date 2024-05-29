import random

from UltraEval.tasks.postprocess import GSM8KPost


def transform(data, num_sample: int, r: random.Random, dataset_name: str):
    text = f"[INST] <<SYS>>\nDu-bist-ein-deutscher-KI-Assistent,-dessen-Aufgabe-es-ist,-den-Benutzern-bei-der-Lösung-ihrer-Probleme-zu-helfen.-Wenn-du-Fragen-beantwortest,-solltest-du-nur-auf-Deutsch-antworten,-damit-die-deutschsprachigen-Benutzer-deine-Antworten-besser-verstehen-können.-Die-Antworten-sollten-so-hilfreich-und-lesbar-wie-möglich-sein-und-keinen-schädlichen-Inhalt-enthalten.-Bitte-achte-darauf,-nicht-auf-Englisch-zu-antworten.\n<</SYS>>\n\n{data['question']} [/INST]"
    correct_answer = data["answer"]
    gsm8kp = GSM8KPost()
    _, processed_correct_answer = gsm8kp([], correct_answer)
    return {
        "input": text,
        "output": correct_answer,
        "processed_output": processed_correct_answer,
    }
