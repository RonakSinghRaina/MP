from youtube_transcript_api import YouTubeTranscriptApi
import json

api = YouTubeTranscriptApi()
transcript = api.fetch('Au1OxVSyGas')
print('Fetched raw:', type(transcript), dir(transcript))
lines = []
for s in transcript:
    # check structure
    text = getattr(s, 'text', str(s))
    start = getattr(s, 'start', 0)
    lines.append(f"{start}s: {text}")

full_text = "\n".join(lines)
with open('video_transcript.txt', 'w', encoding='utf-8') as f:
    f.write(full_text)

print('Transcript saved! Total snippets:', len(lines))
print('Total text length:', len(full_text))
