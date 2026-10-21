import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import snapshot_download

print("Downloading nyrahealth/CrisperWhisper (~3.2 GB)...")
print("Progress bars will appear below. Safe to re-run — resumes automatically.\n")

path = snapshot_download(
    'nyrahealth/CrisperWhisper',
    token=os.getenv('HF_TOKEN'),
    ignore_patterns=['*.msgpack', 'flax_model*', 'tf_model*'],
)

print(f"\nDone! Cached at: {path}")
