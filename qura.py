# =============================================================
# 🌙 Quran AI Tutor v14 — The Listening Ustadh (Senior ML Engineer)
# - Realtime WebRTC (OpenAI Realtime) + Whisper verbose alignment
# - Phoneme-aware Tajweed analysis (via OpenAI models)
# - FAISS retrieval, TTS (OpenAI or gTTS fallback), emotion detection
# =============================================================
# NOTE: Install dependencies separately and set OPENAI_API_KEY in env.
# =============================================================

import os, io, time, json, base64, sqlite3, requests, numpy as np, faiss
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from langdetect import detect
from scipy.spatial.distance import cosine

# Optional DeepFace for emotion (install separately)
DEEPFACE_AVAILABLE = True
try:
    from deepface import DeepFace
    import cv2
except Exception as e:
    print("⚠️ DeepFace/OpenCV not available. Emotion detection disabled. Error:", e)
    DEEPFACE_AVAILABLE = False

# ---------------------------
# OpenAI client
# ---------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")
client = OpenAI(api_key=OPENAI_API_KEY)
print("✅ OpenAI client initialized.")

# ---------------------------
# System prompt (exact user-provided)
# ---------------------------
SYSTEM_PROMPT = (
    "You are a compassionate, wise, and patient Quran teacher (Ustadh) who guides students "
    "in reciting, understanding, and applying the Quran with precision and sincerity. "

    "💫 Core Teaching Role: "
    "Help the learner improve their Quran recitation by analyzing Tajweed accuracy, pronunciation, rhythm, and fluency. "
    "When the student recites aloud, evaluate the audio input with reference to the correct phonetic pattern. "
    "Provide clear, constructive feedback highlighting exact Tajweed errors — e.g., incorrect Makharij (articulation), "
    "missing Ghunnah (nasal sound), weak Qalqalah, or over-shortening of Madd. "

    "🎙️ Voice Feedback Guidelines: "
    "If the recitation is correct, praise the learner warmly (e.g., 'MashaAllah, your articulation of the letter ع is excellent.'). "
    "If the pronunciation is inaccurate, gently correct them by: "
    "1. Identifying the specific letter or sound (e.g., 'The letter ص should be pronounced with a stronger, deeper tone.'). "
    "2. Explaining the Tajweed rule briefly. "
    "3. Giving a phonetic example with transliteration (e.g., 'Try saying “ṣābirīn” with your tongue closer to the top palate.'). "
    "Encourage them to repeat until the sound matches the correct Quranic tone. "

    "📜 Teaching Etiquette: "
    "Always respond with humility, patience, and empathy. Begin with a respectful phrase such as 'Bismillah' or 'Alhamdulillah'. "
    "If the user is struggling or makes repeated mistakes, motivate them gently (e.g., 'Every effort to improve is rewarded by Allah.'). "
    "Never scold, shame, or criticize. Instead, remind them that the Prophet ﷺ said: "
    "‘The one who recites the Qur’an beautifully and precisely will be in the company of the noble angels, "
    "and the one who stumbles while reciting receives double the reward.’ (Sahih Muslim). "

    "📖 Quranic References: "
    "When explaining meaning or lessons, quote directly from the Quran with Surah and Ayah numbers. "
    "Do not interpret beyond your scope — if uncertain, say: 'I am not sure; please consult a qualified scholar.' "
    "Use accurate transliteration (e.g., Surah Al-Fatiha, Ayah 5: 'Iyyaka na'budu wa iyyaka nasta'in'). "
    "Explain meaning in simple, respectful language, and emphasize reflection (Tadabbur). "

    "🕌 Emotional Awareness: "
    "If the user sounds sad, anxious, or demotivated, respond with gentle Quranic comfort. "
    "Use verses about mercy, patience, and hope (e.g., Surah Ash-Sharh 94:5–6: 'Indeed, with hardship comes ease.'). "
    "Encourage spiritual consistency and remind them of Allah’s mercy and reward for sincere effort. "

    "⚙️ Interaction Behavior: "
    "Use short, clear paragraphs. Avoid long lists or academic tone. "
    "When appropriate, summarize progress (e.g., 'Your recitation of Surah Al-Ikhlas improved by 20% accuracy in Madd length.'). "
    "End responses with a gentle encouragement or Du’a (e.g., 'May Allah make your tongue fluent in His Book.'). "

    "🛑 Safety & Ethics: "
    "Never generate or promote anything disrespectful to the Quran or Islamic teachings. "
    "Do not interpret Sharia rulings, theology, or sensitive topics. "
    "Focus only on Tajweed, pronunciation, and accurate Quranic recitation guidance. "
    "If a question falls outside your expertise, say 'I am not sure; please consult a scholar.' "

    "Your tone should always reflect warmth, patience, sincerity, and scholarly respect — like a real Ustadh guiding a beloved student. "
)

# ---------------------------
# SQLite memory
# ---------------------------
DB_PATH = "/content/quran_tutor_v14.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    query TEXT,
    answer TEXT,
    verses TEXT,
    score REAL DEFAULT 0,
    emotion TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def save_memory(user_id, query, answer, verses, score=0.0, emotion="neutral"):
    conn.execute(
        "INSERT INTO memory (user_id, query, answer, verses, score, emotion) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, query, answer, json.dumps(verses), float(score), emotion)
    )
    conn.commit()

# ---------------------------
# Load Quran dataset
# ---------------------------
print("📖 Loading Quran dataset ...")
quran = requests.get("https://api.alquran.cloud/v1/quran/ar.alafasy").json()
verses = []
for surah in quran["data"]["surahs"]:
    for ayah in surah["ayahs"]:
        verses.append({
            "id": f"{surah['number']}:{ayah['numberInSurah']}",
            "surah": surah["englishName"],
            "text": ayah["text"]
        })
print(f"✅ Loaded {len(verses)} verses.")

# ---------------------------
# Embeddings + FAISS (ensure float32 contiguous)
# ---------------------------
EMB_PATH = "/content/quran_embeddings_v14.npz"

def build_embeddings():
    print("⚙️ Building embeddings for Quran verses (first run)...")
    texts = [v["text"] for v in verses]
    vectors = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        resp = client.embeddings.create(model="text-embedding-3-large", input=batch)
        vectors.extend([d.embedding for d in resp.data])
    np.savez(EMB_PATH, vectors=np.array(vectors, dtype="float32"))
    return np.array(vectors, dtype="float32")

def load_or_build_embeddings():
    if os.path.exists(EMB_PATH):
        data = np.load(EMB_PATH)
        vec = data["vectors"]
    else:
        vec = build_embeddings()
    return np.ascontiguousarray(vec, dtype="float32")

vectors = load_or_build_embeddings()
vectors = np.ascontiguousarray(vectors, dtype="float32")

index = faiss.IndexFlatIP(vectors.shape[1])
faiss.normalize_L2(vectors)
index.add(vectors)
print(f"✅ FAISS index built with {index.ntotal} embeddings.")

# ---------------------------
# Retrieval helper
# ---------------------------
def retrieve(query, top_k=5, min_score=0.25):
    try:
        q_emb = client.embeddings.create(model="text-embedding-3-large", input=query).data[0].embedding
        q_emb = np.ascontiguousarray(np.array(q_emb, dtype="float32").reshape(1, -1))
        faiss.normalize_L2(q_emb)
        D, I = index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if score >= min_score:
                results.append({
                    "score": float(score),
                    "verse": verses[idx]["text"],
                    "id": verses[idx]["id"],
                    "surah": verses[idx]["surah"]
                })
        return results
    except Exception as e:
        print("Retrieval error:", e)
        return []

# ---------------------------
# Emotion detection helper
# ---------------------------
def detect_emotion_from_b64(img_b64):
    if not DEEPFACE_AVAILABLE:
        return "neutral"
    try:
        img_bytes = base64.b64decode(img_b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return "neutral"
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list): result = result[0]
        return result.get("dominant_emotion", "neutral").lower()
    except Exception as e:
        print("DeepFace detect error:", e)
        return "neutral"

def map_emotion_to_tone(emotion_label):
    e = (emotion_label or "neutral").lower()
    if e in ("happy", "joy", "smile"): return "cheerful"
    if e in ("sad", "angry", "upset", "fear"): return "comforting"
    if e in ("surprise", "surprised", "confused"): return "encouraging"
    return "neutral"

# ---------------------------
# Whisper transcription (verbose)
# ---------------------------
def transcribe_verbose(audio_bytes):
    tmp = f"/tmp/trans_v_{int(time.time()*1000)}.wav"
    with open(tmp, "wb") as f: f.write(audio_bytes)
    with open(tmp, "rb") as af:
        # verbose JSON returns word/segment timestamps (useful for alignment)
        resp = client.audio.transcriptions.create(model="whisper-1", file=af, response_format="verbose_json")
    # resp may be a dict-like; return raw dict
    return resp if isinstance(resp, dict) else json.loads(resp)

# ---------------------------
# Phoneme-aware Tajweed evaluation (OpenAI-only flow)
#  - Step 1: Whisper verbose -> segments with timestamps
#  - Step 2: Ask GPT (gpt-4o) to perform phoneme-level comparison + rule detection
#    (we rely on GPT's linguistic knowledge and Whisper timestamps; this keeps everything OpenAI-powered)
# ---------------------------
def tajweed_phoneme_analysis(target_text, audio_bytes):
    # 1) Get verbose speech->words (+ timestamps)
    verbose = transcribe_verbose(audio_bytes)
    # Build a compact representation for the model: segments with text + start/end
    segments = []
    if "segments" in verbose:
        for s in verbose["segments"]:
            seg_text = s.get("text", "").strip()
            start = s.get("start", 0.0)
            end = s.get("end", 0.0)
            segments.append({"text": seg_text, "start": start, "end": end})
    else:
        # fallback: use top-level text field
        segments = [{"text": verbose.get("text", ""), "start": 0.0, "end": 0.0}]

    # 2) Compose a thorough instruction for GPT to analyze pronunciation & Tajweed
    analysis_prompt = {
        "instruction": (
            "You are an expert Tajweed teacher. Compare the student's recitation (provided as segmented transcript) "
            "against the target Quranic verse. Use the segments array which contains timestamps and the student's spoken text.\n\n"
            "Produce a strict JSON object with fields:\n"
            " - score: integer 0-100 (overall recitation accuracy)\n"
            " - errors: array of {rule: 'Madd/Ghunnah/Qalqalah/Makharij/...', span: 'word or phoneme', description: 'short'}\n"
            " - student_text: the joined student transcription\n"
            " - feedback_ar: short Arabic teacher feedback (1-2 sentences)\n"
            " - feedback_en: short transliteration + practical exercise (1-2 sentences)\n\n"
            "Be concise, kind, and follow the SYSTEM_PROMPT teaching etiquette. Use the segments strictly to reference timing if relevant."
        ),
        "target": target_text,
        "segments_sample": segments[:30]  # keep prompt size reasonable
    }

    # 3) Call GPT to analyze and return JSON
    user_msg = json.dumps(analysis_prompt, ensure_ascii=False)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.08,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}
            ],
        )
        content = resp.choices[0].message.content.strip()
        # Model should return JSON; try parsing
        try:
            parsed = json.loads(content)
        except Exception:
            # ask model to reformat strictly as JSON
            re_req = client.chat.completions.create(
                model="gpt-4o",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "Reformat the previous assistant reply strictly as JSON."},
                    {"role": "user", "content": content}
                ],
            )
            parsed = json.loads(re_req.choices[0].message.content.strip())
        # ensure keys present
        parsed.setdefault("score", 0)
        parsed.setdefault("errors", [])
        parsed.setdefault("student_text", " ".join([s["text"] for s in segments]))
        parsed.setdefault("feedback_ar", "")
        parsed.setdefault("feedback_en", "")
        return parsed
    except Exception as e:
        print("Tajweed analysis error:", e)
        # fallback: similarity-based rough score
        joined = " ".join([s["text"] for s in segments])
        sim = 0.0
        try:
            ta = client.embeddings.create(model="text-embedding-3-large", input=target_text).data[0].embedding
            tb = client.embeddings.create(model="text-embedding-3-large", input=joined).data[0].embedding
            sim = 1 - cosine(np.array(ta), np.array(tb))
            sim = float(max(0.0, min(sim, 1.0)))
        except Exception:
            sim = 0.0
        return {
            "score": int(sim * 100),
            "errors": [],
            "student_text": joined,
            "feedback_ar": "ملاحظات عامة: واصل التدريب على مخارج الحروف.",
            "feedback_en": "General note: keep practicing letter articulation."
        }

# ---------------------------
# TTS (OpenAI realtime or fallback)
# ---------------------------
def make_tts_base64(text, lang="ar"):
    try:
        speech = client.audio.speech.create(model="gpt-4o-mini-tts", voice="alloy", input=text)
        audio_bytes = speech.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print("Realtime TTS failed:", e)
        try:
            from gtts import gTTS
            tmp = f"/tmp/gtts_{int(time.time()*1000)}.mp3"
            gTTS(text=text, lang=lang).save(tmp)
            with open(tmp, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e2:
            print("gTTS fallback failed:", e2)
            return None

# ---------------------------
# Web server (Flask)
# ---------------------------
from pyngrok import ngrok
from flask import Flask, request, jsonify
app = Flask(__name__)
CORS(app)

# Basic ask endpoint (text)
@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    user_id = data.get("user_id", "guest")
    query = data.get("query", "").strip()
    image_b64 = data.get("image", None)

    if not query:
        return jsonify({"error": "empty query"}), 400

    emotion = detect_emotion_from_b64(image_b64) if image_b64 else "neutral"
    context = retrieve(query)
    verses_text = "\n".join([f"{v['id']} ({v['surah']}): {v['verse']}" for v in context])
    prompt = f"{SYSTEM_PROMPT}\nStudent: {query}\nEmotion: {emotion}\nContext:\n{verses_text}\nAnswer concisely."

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.18,
            messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":prompt}]
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    tts_b64 = make_tts_base64(answer, lang="ar")
    save_memory(user_id, query, answer, context, score=0.0, emotion=emotion)
    return jsonify({"answer": answer, "verses": context, "audio": tts_b64, "emotion": emotion})

# Tajweed endpoint (multipart form: audio file + verse_id or reference)
@app.route("/tajweed", methods=["POST"])
def tajweed_route():
    try:
        user_id = request.form.get("user_id", "guest")
        reference = request.form.get("reference", None)
        verse_id = request.form.get("verse_id", None)

        if reference:
            target_text = reference
        elif verse_id:
            found = next((v for v in verses if v["id"] == verse_id), None)
            if not found:
                return jsonify({"error": "verse_id not found"}), 404
            target_text = found["text"]
        else:
            return jsonify({"error": "reference or verse_id required"}), 400

        if "audio" not in request.files:
            return jsonify({"error": "audio file required"}), 400

        audio_bytes = request.files["audio"].read()
        analysis = tajweed_phoneme_analysis(target_text, audio_bytes)
        audio_feedback = make_tts_base64(analysis.get("feedback_ar", ""), lang="ar") if analysis.get("feedback_ar") else None

        save_memory(user_id, f"tajweed_{verse_id or reference[:24]}", json.dumps(analysis), [target_text], score=analysis.get("score", 0), emotion="neutral")
        return jsonify({
            "reference": target_text,
            "student_text": analysis.get("student_text"),
            "score_percent": analysis.get("score"),
            "errors": analysis.get("errors"),
            "feedback_ar": analysis.get("feedback_ar"),
            "feedback_en": analysis.get("feedback_en"),
            "audio_feedback": audio_feedback
        })
    except Exception as e:
        print("tajweed route error:", e)
        return jsonify({"error": str(e)}), 500

# ---------------------------
# WebRTC offer handler (forward SDP to OpenAI Realtime)
# - Your PHP frontend should POST { "sdp": "<offer>" } to this endpoint.
# - We forward the SDP to OpenAI Realtime endpoint and return the SDP answer.
# ---------------------------
@app.route("/webrtc/offer", methods=["POST"])
def webrtc_offer():
    data = request.get_json(force=True)
    sdp_offer = data.get("sdp")
    if not sdp_offer:
        return jsonify({"error": "SDP offer required"}), 400

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/sdp",
    }

    # Use realtime model and enable TTS streaming (voice param optional)
    params = "?model=gpt-4o-realtime-preview&voice=alloy"
    try:
        resp = requests.post(f"https://api.openai.com/v1/realtime{params}", headers=headers, data=sdp_offer.encode("utf-8"), timeout=30)
        # return SDP answer as text with correct content type
        return resp.text, 200, {"Content-Type": "application/sdp"}
    except Exception as e:
        print("webrtc/offer error:", e)
        return jsonify({"error": str(e)}), 500

# ---------------------------
# Misc endpoints
# ---------------------------
@app.route("/lesson", methods=["GET"])
def lesson():
    level = request.args.get("level", "beginner")
    topics = {
        "beginner": ["Al-Fatiha meaning", "Makharij basics", "Qalqalah rules"],
        "intermediate": ["Rules of Noon Sakinah", "Madd types", "Tafsir Al-Kahf"],
        "advanced": ["Balaghah analysis", "Ulum Al-Quran", "Deep Tafsir discussion"]
    }
    return jsonify({"level": level, "topics": topics.get(level, [])})

@app.route("/session_status", methods=["GET"])
def session_status():
    return jsonify({"session_id": "default", "attention": 0.94, "mood": "focused"})

# ---------------------------
# Launch (ngrok for dev)
# ---------------------------
from pyngrok import ngrok as _ngrok
public_url = _ngrok.connect(5000).public_url
print(f"🌍 Quran AI Tutor v14 running at {public_url}")
app.run(host="0.0.0.0", port=5000)

