## 0. Installing Huggin Face's Transformers and Loading Dependencies
"""



"""## 1. Data Preprocessing: Extracting transcript from a youtube video url"""


from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

summarizer = pipeline("summarization")

"""### 1.1 Getting the video transcript"""

def get_transcript(video):
  video_id = video.split("=")[1]
  transcript = YouTubeTranscriptApi.get_transcript(video_id)
  return transcript

"""### 1.2 Preprocessing the transcript"""

def preprocess_transcript(transcript):
  doc = []
  for line in transcript:
    if "\n" in line['text']:
      x = line["text"].replace("\n", " ")
      doc.append(x)
    else:
      doc.append(line['text'])

  paragraph = " ".join(doc)
  paragraph = paragraph.replace(".", ".<eos>")
  paragraph = paragraph.replace("!", "!<eos>")
  paragraph = paragraph.replace("?", "?<eos>")
  sentences = paragraph.split("<eos>")
  return sentences

"""### 1.3 Chunking the data, since the text is too long"""

def chunk_sentences(sentences):
  max_chunk = 500
  curr_chunk = 0
  chunks = []
  for sentence in sentences:
      words = sentence.split(" ")
      if chunks and len(chunks[curr_chunk]) + len(words) <= max_chunk:
          chunks[curr_chunk].extend(words)
      else:
          i = 0
          while i < len(words):
              chunk_size = min(max_chunk, len(words) - i)
              chunks.append(words[i:i + chunk_size])
              i += chunk_size
              curr_chunk = len(chunks) - 1
  for chunk_id in range(len(chunks)):
    chunks[chunk_id] = " ".join(chunks[chunk_id])
  return chunks

"""## 2. Generating the summary"""

def get_summary(chunks):
  res = summarizer(chunks, max_length=30, min_length=10, do_sample=False)
  finalSummary = " ".join([summary["summary_text"] for summary in res])
  return finalSummary