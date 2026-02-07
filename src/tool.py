import pandas as pd
from zstandard import ZstdDecompressor
import json
import io
files_target = [
    "data/ppd/Postpartum_Depression_comments.zst", 
    "data/ppd/Postpartum_Depression_submissions.zst", 
    "data/ppd/postpartumprogress_comments.zst", 
    "data/ppd/postpartumprogress_submissions.zst"
]

dcts = ZstdDecompressor()
data = []
for file in files_target:
    with open(file, "rb") as f:
        with dcts.stream_reader(f) as reader:
            lines_raw = io.TextIOWrapper(reader, encoding="utf-8")
            for lines in lines_raw:
                line = json.loads(lines)
                post_info = {
                    'selftext': line.get('selftext', '') or line.get('body', ''),
                    'subreddit': 'postpartum_depression',
                    'id': line.get('id')
                }
                if post_info['selftext'] != '[deleted]' and post_info['selftext'] != '[removed]' and len(post_info['selftext']) > 10:
                    data.append(post_info)

df = pd.DataFrame(data)
df.to_parquet("data/postpartum.parquet")