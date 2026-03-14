from flask import Flask, request, jsonify
from flask_cors import CORS
from mistralai.client import Mistral
import os
import hashlib
import time

app = Flask(__name__)
CORS(app)

API_KEY = os.environ.get("MISTRAL_API_KEY", "")
client = Mistral(api_key=API_KEY)
MODEL = "mistral-large-latest"

translation_cache = {}
CACHE_DURATION = 86400

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


@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data['text']
    target_lang = data['target_language']
    source_lang = data.get('source_language', 'auto')

    if target_lang not in LANGUAGES:
        return jsonify({"error": f"Lingua '{target_lang}' non supportata"}), 400

    if not text.strip():
        return jsonify({"translated_text": ""})

    cache_key = hashlib.md5(f"{text}:{target_lang}".encode()).hexdigest()
    if cache_key in translation_cache:
        cached = translation_cache[cache_key]
        if time.time() - cached['time'] < CACHE_DURATION:
            return jsonify({"translated_text": cached['text'], "cached": True})

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

    translated_text = response.choices[0].message.content.strip()
    translation_cache[cache_key] = {'text': translated_text, 'time': time.time()}

    return jsonify({"translated_text": translated_text})


@app.route('/translate-batch', methods=['POST'])
def translate_batch():
    data = request.json
    texts = data.get('texts', [])
    target_lang = data['target_language']

    if not texts or target_lang not in LANGUAGES:
        return jsonify({"error": "Parametri non validi"}), 400

    separator = "\n|||SEPARATOR|||\n"
    combined = separator.join(texts)
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

    return jsonify({"translated_texts": translated_texts[:len(texts)]})


@app.route('/languages', methods=['GET'])
def languages():
    return jsonify({"languages": LANGUAGES, "total": len(LANGUAGES)})


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "service": "LumiBear Translate API",
        "model": MODEL,
        "languages": len(LANGUAGES),
        "powered_by": "Mistral AI + Claude (Anthropic)"
    })
