import re
import os
import ssl
from tqdm import tqdm
from gtts import gTTS
from rapidfuzz import fuzz
from pydub import AudioSegment
import speech_recognition as sr
from transformers import AutoTokenizer
from deep_translator import GoogleTranslator
from arabert.preprocess import ArabertPreprocessor

ssl._create_default_https_context = ssl._create_unverified_context


tokenizers = [
    "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "Mhassanen/nllb-200-600M-En-Ar",
    "Ammar-alhaj-ali/arabic-MARBERT-poetry-classification",
    "Ebtihal/AraBertMo_base_V10",
    "abdusah/arabert-ner",
    "bakrianoo/t5-arabic-large",
    "asafaya/bert-large-arabic",
    "ychenNLP/arabic-ner-ace",
    "MIIB-NLP/Arabic-question-generation",
    "Hezam/ArabicT5-news-classification-generation",
    "malmarjeh/transformer",
]

arabert_prep = ArabertPreprocessor(model_name="aubmindlab/bert-base-arabertv2")

arabic_pattern = re.compile(r"[\u0600-\u06FF]+")


def process_token(token):
    token = re.sub(r'[^\w\s]', '', token.replace('#',''))
    return token if arabic_pattern.search(token) else None

tokens_dict = list({token for tokenizer in tokenizers for token in AutoTokenizer.from_pretrained(tokenizer).get_vocab() if (processed_token := process_token(token))})


class Word():
    Arabic = 0
    English = 1
    Unknown = 2


def ar_text_to_speech_to_en_text(text):
    tts = gTTS(text=text, lang='ar')
    mp3_filename = "temp_speech.mp3"
    wav_filename = "temp_speech.wav"
    tts.save(mp3_filename)

    AudioSegment.from_mp3(mp3_filename).export(wav_filename, format="wav")
    os.remove(mp3_filename) 

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        print("Error from recognizer")
        return None

def which(text: str) -> str:
    try:
        text1 = ar_text_to_speech_to_en_text(text)
        if text1 is None:
            return Word.Unknown
        text2 = GoogleTranslator(source='ar', target='en').translate(text)
        similarity = fuzz.ratio(text1.lower(), text2.lower())
    except:
        print("Error from translator")
        return Word.Unknown
    return Word.Arabic if similarity < 70 else Word.English


ar, en, uk = 0, 0, 0
ar_token, en_token, unknown_token = [], [], []
with tqdm(tokens_dict) as tqdm_token:
    for token in tqdm_token:
        state = which(token)
        if state == Word.Arabic:
            ar_token.append(token)
            ar += 1
        elif state == Word.English:
            en_token.append(token)
            en += 1
        else:
            unknown_token.append(token)
            uk += 1
        tqdm_token.set_postfix(ar=ar, en=en, unknown=uk)

os.makedirs("tokens", exist_ok=True)

for name in ["ar_tokens", "en_token", "unknown_token"]:
    with open(f"tokens/{name}.txt", "w") as f:
        f.writelines(eval(f"{name}"))


