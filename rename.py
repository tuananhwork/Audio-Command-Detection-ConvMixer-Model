#!/usr/bin/env python3
import re
from pathlib import Path
from collections import defaultdict

RAW_DIR = Path("data/raw")
DRY_RUN = False  # đổi thành False để rename thật

pattern = re.compile(r"^(.+?)_speaker(\d+)_(\d+)\.wav$", re.IGNORECASE)

for folder in RAW_DIR.iterdir():
    if not folder.is_dir():
        continue

    files = sorted(folder.glob("*.wav"))
    speaker_groups = defaultdict(list)

    # Group file theo speaker
    for file in files:
        m = pattern.match(file.name)
        if not m:
            print(f"[SKIP] {file}")
            continue

        speaker = m.group(2)
        old_index = int(m.group(3))
        speaker_groups[speaker].append((old_index, file))

    print(f"\n=== {folder.name} ===")

    # Với mỗi speaker, sort theo index cũ rồi đánh lại từ 001
    for speaker in sorted(speaker_groups.keys(), key=lambda x: int(x)):
        speaker_files = sorted(speaker_groups[speaker], key=lambda x: x[0])

        for new_idx, (_, file) in enumerate(speaker_files, start=1):
            new_name = f"{folder.name}_speaker{speaker}_{new_idx:03d}.wav"
            new_path = file.with_name(new_name)

            print(f"{file.name} -> {new_name}")

            if not DRY_RUN:
                if new_path.exists() and new_path != file:
                    raise FileExistsError(f"Target already exists: {new_path}")
                file.rename(new_path)
