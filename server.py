from flask import Flask, request, jsonify
from flask_cors import CORS
from mistralai.client import Mistral
import os
import hashlib
import json
import time

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("MISTRAL_API_KEY", "")
client = Mistral(api_key=API_KEY)
MODEL = "mistral-large-latest"

CACHE_DIR = "/tmp/lumibear_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

LANGUAGES = {
    "it": "Italian", "en": "English", "fr": "French",
    "de": "German", "es": "Spanish", "pt": "Portuguese",
    "nl": "Dutch", "da": "Danish", "sv": "Swedish",
    "no": "Norwegian", "fi": "Finnish", "is": "Icelandic",
    "ca": "Catalan", "eu": "Basque", "mt": "Maltese",
    "pl": "Polish", "cs": "Czech", "sk": "Slovak",
    "hu": "Hungarian", "ro": "Romanian", "bg": "Bulgarian",
    "hr": "Croatian", "sr": "Serbian", "sl": "Slovenian",
    "sq": "Albanian", "el": "Greek", "lt": "Lithuanian",
    "lv": "Latvian", "et": "Estonian",
    "ru": "Russian", "uk": "Ukrainian", "ka": "Georgian",
    "az": "Azerbaijani", "kk": "Kazakh", "mn": "Mongolian",
    "hy": "Armenian",
    "ar": "Arabic", "he": "Hebrew", "fa": "Persian",
    "tr": "Turkish", "ku": "Kurdish",
    "hi": "Hindi", "bn": "Bengali", "ur": "Urdu",
    "ta": "Tamil", "te": "Telugu", "ne": "Nepali", "si": "Sinhala",
    "lo": "Lao", "th": "Thai", "vi": "Vietnamese",
    "km": "Khmer", "my": "Burmese", "ms": "Malay",
    "id": "Indonesian", "tl": "Filipino/Tagalog",
    "zh": "Chinese (Simplified)", "zh-TW": "Chinese (Traditional)",
    "ja": "Japanese", "ko": "Korean",
    "sw": "Swahili", "am": "Amharic", "ha": "Hausa",
    "yo": "Yoruba", "zu": "Zulu", "af": "Afrikaans", "so": "Somali",
    "ga": "Irish", "cy": "Welsh", "la": "Latin"
}

GLOSSARY_NO_TRANSLATE = [
    "LumiBear", "GR-QUANTUM", "Quantum Crisis", "Quantum Engine",
    "Vishapakar", "Clausola di Luce", "Nodo Sacro", "CDSB", "DVP", "Deneb"
]


def get_cache(text, target_lang):
    cache_key = hashlib.md5(f"{text}:{target_lang}".encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)['text']
        except:
            pass
    return None


def set_cache(text, target_lang, translated):
    cache_key = hashlib.md5(f"{text}:{target_lang}".encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, f"{cache_key}.json")
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({'text': translated, 'time': time.time()}, f, ensure_ascii=False)
    except:
        pass


@app.route('/translate', methods=['POST'])
def translate():
    try:
        data = request.json
        text = data['text']
        target_lang = data['target_language']

        if target_lang not in LANGUAGES:
            return jsonify({"error": f"Lingua '{target_lang}' non supportata"}), 400

        if not text.strip():
            return jsonify({"translated_text": ""})

        cached = get_cache(text, target_lang)
        if cached:
            return jsonify({"translated_text": cached, "cached": True})

        terms = ", ".join(GLOSSARY_NO_TRANSLATE)
        lang_name = LANGUAGES[target_lang]

        response = client.chat.complete(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Traduci il seguente testo in {lang_name}. "
                        "Restituisci SOLO la traduzione, senza commenti. "
                        "Mantieni la formattazione HTML se presente. "
                        f"NON tradurre questi termini: {terms}"
                    )
                },
                {"role": "user", "content": text}
            ],
            temperature=0.1
        )

        translated = response.choices[0].message.content.strip()
        set_cache(text, target_lang, translated)
        return jsonify({"translated_text": translated, "cached": False})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/translate-batch', methods=['POST'])
def translate_batch():
    try:
        data = request.json
        texts = data.get('texts', [])
        target_lang = data['target_language']

        if not texts or target_lang not in LANGUAGES:
            return jsonify({"error": "Parametri non validi"}), 400

        results = [None] * len(texts)
        to_translate = []
        to_translate_idx = []

        for i, text in enumerate(texts):
            if not text.strip():
                results[i] = ""
                continue
            cached = get_cache(text, target_lang)
            if cached:
                results[i] = cached
            else:
                to_translate.append(text)
                to_translate_idx.append(i)

        if to_translate:
            separator = "\n|||SEPARATOR|||\n"
            combined = separator.join(to_translate)
            lang_name = LANGUAGES[target_lang]
            terms = ", ".join(GLOSSARY_NO_TRANSLATE)

            response = client.chat.complete(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Traduci i seguenti testi in {lang_name}. "
                            "Separati da |||SEPARATOR|||. "
                            "Mantieni lo stesso separatore. "
                            "Restituisci SOLO le traduzioni. "
                            f"NON tradurre: {terms}"
                        )
                    },
                    {"role": "user", "content": combined}
                ],
                temperature=0.1
            )

            result = response.choices[0].message.content.strip()
            translated_texts = [t.strip() for t in result.split("|||SEPARATOR|||")]

            for j, idx in enumerate(to_translate_idx):
                if j < len(translated_texts):
                    results[idx] = translated_texts[j]
                    set_cache(to_translate[j], target_lang, translated_texts[j])
                else:
                    results[idx] = to_translate[j]

        return jsonify({"translated_texts": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/languages', methods=['GET'])
def languages():
    return jsonify({"languages": LANGUAGES, "total": len(LANGUAGES)})


@app.route('/health', methods=['GET'])
def health():
    cache_count = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    return jsonify({
        "status": "ok",
        "service": "LumiBear Translate API",
        "model": MODEL,
        "languages": len(LANGUAGES),
        "cached_translations": cache_count,
        "powered_by": "Mistral AI + Claude (Anthropic)"
    })


@app.route('/cache-stats', methods=['GET'])
def cache_stats():
    cache_count = len([f for f in os.listdir(CACHE_DIR) if f.endswith('.json')])
    return jsonify({
        "cached_translations": cache_count,
        "cache_dir": CACHE_DIR
    })
