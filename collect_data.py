import re
import os
import ssl
import shutil
import argparse
from tqdm import tqdm
from gtts import gTTS
import huggingface_hub
from rapidfuzz import fuzz
from pydub import AudioSegment
import speech_recognition as sr
from huggingface_hub import HfApi
from utils import translate_if_needed
from pytubefix import YouTube, Channel
from transformers import AutoTokenizer
from deep_translator import GoogleTranslator
from arabert.preprocess import ArabertPreprocessor
from datasets import Dataset, Audio, Features, Value
from youtube_transcript_api import YouTubeTranscriptApi


ssl._create_default_https_context = ssl._create_unverified_context

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process YouTube channels for audio and captions.")
    parser.add_argument('--urls_file', type=str, default='channels.txt', help='File containing YouTube channel URLs')
    parser.add_argument('--hub_dataset_name', type=str, default='<USERNAME/REPO>', help='Base name of the dataset on Hugging Face Hub')
    parser.add_argument('--private', action='store_true', help='Push dataset as private')
    parser.add_argument('--hf_token', type=str, default='<HUGGINGFACE_API_TOKEN>', help='Hugging Face authentication token')
    return parser.parse_args()

args = parse_arguments()

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
        audio = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        audio = None
        
    os.remove(wav_filename)
    return audio

def translate_if_needed(text: str) -> str:
    text1 = ar_text_to_speech_to_en_text(text)
    if text1 is None:
        return text
    text2 = GoogleTranslator(source='ar', target='en').translate(text)

    similarity = fuzz.ratio(text1.lower(), text2.lower())

    return text if similarity < 70 else text2

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
english_pattern = re.compile(r"[a-zA-Z]+")


def process_token(token):
    token = re.sub(r'[^\w\s]', '', token.replace('#',''))
    return token if arabic_pattern.search(token) else None

tokens_dict = list({token for tokenizer in tokenizers for token in AutoTokenizer.from_pretrained(tokenizer).get_vocab() if (processed_token := process_token(token))})


def detect_and_replace_transliterated_words(arabic_text):
    arabic_text = arabert_prep.preprocess(arabic_text)
    tokens = arabic_text.split()
    
    temp = []
    processed_tokens = []
    for token in tokens:
        token_cleaned = re.sub(r'[^\w\s]', '', token)  # Clean punctuation
        if token_cleaned.lower() not in tokens_dict:
            temp.append(token)
        else:
            if temp:
                processed_tokens.append(translate_if_needed(arabert_prep.unpreprocess(' '.join(temp))))
                temp.clear()
            processed_tokens.append(token)
            temp.clear()

    if temp:
        processed_tokens.append(translate_if_needed(arabert_prep.unpreprocess(' '.join(temp))))
        temp.clear()

    return arabert_prep.unpreprocess(' '.join(processed_tokens))


def get_captions(video_id):
    manual = None
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            transcript = transcript_list.find_manually_created_transcript(["ar", "en"])
            caption_data = transcript.fetch()
            manual = True
        except:
            try:
                transcript = transcript_list.find_generated_transcript(["ar", "en"])
                caption_data = transcript.fetch()
                for caption in caption_data:
                    try:
                        caption["text"] = detect_and_replace_transliterated_words(caption["text"])
                    except:
                        pass
                manual = False
            except:
                return None, manual
    except Exception:
        return None, manual
    arabic_and_english_captions = [
        caption for caption in caption_data 
        if arabic_pattern.search(caption["text"]) and english_pattern.search(caption["text"])
    ]
    
    return (arabic_and_english_captions if arabic_and_english_captions else None, manual)


def cut_audio(audio_path, captions, manual):
    audio = AudioSegment.from_file(audio_path)
    audio_chunks = []

    if manual:
        # Manual captions: Use start and end times directly
        for caption in captions:
            start = caption["start"]
            duration = caption["duration"]
            end = start + duration
            # Clip the audio segment
            audio_chunk = audio[start * 1000 : min(end * 1000, len(audio))]
            audio_chunks.append(audio_chunk)
    else:
        # Auto-generated captions: Handle potential overlaps and gaps
        for i in range(len(captions)):
            start = captions[i]["start"]
            if i + 1 < len(captions):
                end = captions[i + 1]["start"]
            else:
                duration = captions[i]["duration"]
                end = start + duration
            
            # Ensure the end time does not exceed the audio length
            end = int(min(end, len(audio) / 1000))
            
            # Clip the audio segment
            audio_chunk = audio[start * 1000 : end * 1000]
            audio_chunks.append(audio_chunk)
    
    return audio_chunks


# Load URLs from file
with open(args.urls_file, 'r') as file:
    urls = [line.strip() for line in file]
urls = list(set(urls))

# Authenticate with Hugging Face Hub
huggingface_hub.login(token=args.hf_token)

os.makedirs("audio", exist_ok=True)

batch_size = 100_000  # Number of records to process in one batch
batch_data = {"audio": [], "text": [], "manual": []}
parquet_counter = 0
repo_index = 1

# Create initial repository
hub_dataset_name = f"{args.hub_dataset_name}-{repo_index}"
api = HfApi()
repo_url = api.create_repo(repo_id=hub_dataset_name, token=args.hf_token, repo_type="dataset", private=args.private)
print(f"Created repository: {repo_url}")

for url in tqdm(urls):
    channel = Channel(url)
    with tqdm(channel.videos, leave=True) as tqdm_videos:
        for video in tqdm_videos:
            tqdm_videos.set_postfix(video=video.video_id)
            subtitles, manual = get_captions(video.video_id)
            if subtitles:
                tqdm_videos.set_postfix(video=video.video_id, manual=manual, chunks=len(subtitles))
                try:
                    yt = YouTube(f"https://www.youtube.com/watch?v={video.video_id}")
                    audio_stream = yt.streams.filter(only_audio=True).first()
                    audio_path = audio_stream.download(output_path=str("audio"), filename=f"full_audio.mp4")

                    audio_chunks = cut_audio(audio_path, subtitles, manual)
                    for chunk, caption in zip(audio_chunks, subtitles):
                        chunk.export(f"audio/{video.video_id}_{caption['start']}.wav", format="wav")
                        batch_data["audio"].append(f"audio/{video.video_id}_{caption['start']}.wav")
                        batch_data["text"].append(caption["text"].strip().strip('"').strip("'"))
                        batch_data["manual"].append(manual)

                    if len(batch_data["audio"]) >= batch_size:
                        tqdm_videos.set_postfix(video=video.video_id, manual=manual, chunks=len(subtitles), uploading=len(batch_data["audio"]))
                        new_data = Dataset.from_dict(batch_data, features=Features({"audio": Audio(sampling_rate=16000), "text": Value("string"), "manual": Value("bool")}))
                        new_data.push_to_hub(hub_dataset_name, private=args.private)
                        batch_data = {"audio": [], "text": [], "manual": []}
                        parquet_counter += 1
                        if os.path.exists("audio"):
                            shutil.rmtree("audio")
                            os.makedirs("audio")
                        
                        # Increment repository index and create a new repo
                        repo_index += 1
                        hub_dataset_name = f"{args.hub_dataset_name}-{repo_index}"
                        repo_url = api.create_repo(repo_id=hub_dataset_name, token=args.hf_token, repo_type="dataset", private=args.private)
                        print(f"Created repository: {repo_url}")

                except Exception as e:
                    print(f"Failed with {video.video_id}: {e}")
                    continue

# Push any remaining data in the batch
if batch_data["audio"]:
    new_data = Dataset.from_dict(batch_data, features={"audio": Audio(sampling_rate=16000), "text": Value("string"), "manual": Value("bool")})
    new_data.push_to_hub(hub_dataset_name, private=args.private)

print("\n\n All data have been pushed successfully !!!\n\n")
