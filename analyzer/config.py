# analyzer/config.py
TRANSCRIBE_LANG = None  # for Hinglish in Roman script
CHUNK_DURATION_MIN = 15  # minutes per chunk
HOST_AUDIO_FILENAME = "sanidhya.wav"
GUEST_AUDIO_FILENAME = "guest.wav"
EXTRACTION_CATEGORIES = [
    "unique-facts",
    "one-liners",
    "funny",
    "intellectual",
    "high-energy",
    "relatable",
    "controversial",
]
# EXTRACTION_CATEGORIES = [
#     "emotional", "intellectual", "vulnerable", "high-energy",
#     "funny", "controversial", "relatable", "mic-drop",
#     "nostalgic", "philosophical", "anecdotal", "transformational",
#     "wtf", "expertise", "wholesome", "cultural", "behind-the-scenes", "cliffhanger"
# ]
GPT_MODEL = "gpt-4.1-mini"
WHISPER_MODEL = "whisper"
API_PRICING = {GPT_MODEL: 0.001, WHISPER_MODEL: 0.006}
FRAME_RATE = 30
