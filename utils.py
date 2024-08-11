import os
from gtts import gTTS
from rapidfuzz import fuzz
from pydub import AudioSegment
import speech_recognition as sr
from deep_translator import GoogleTranslator

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
        return None

def translate_if_needed(text: str) -> str:
    text1 = ar_text_to_speech_to_en_text(text)
    if text1 is None:
        return text
    text2 = GoogleTranslator(source='ar', target='en').translate(text)

    similarity = fuzz.ratio(text1.lower(), text2.lower())

    return text if similarity < 70 else text2

if __name__ == '__main__':
    # Example usage
    text = "سكول"  # Arabic text
    word = translate_if_needed(text)
    print(f"word is: {word}")