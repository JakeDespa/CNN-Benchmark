import hashlib
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent

splits = {
    'train': BASE_DIR / 'train-20261007T150600Z-1-001' / 'train',
    'val':   BASE_DIR / 'val-20261007T145448Z-1-001' / 'val',
    'test':  BASE_DIR / 'test-20261007T152143Z-1-001' / 'test'
}

def dhash(image, hash_size=8):
    """Compute difference hash (dHash) for perceptual image similarity."""
    img = image.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = list(img.getdata())
    diff = []
    for row in range(hash_size):
        for col in range(hash_size):
            left = pixels[row * (hash_size + 1) + col]
            right = pixels[row * (hash_size + 1) + col + 1]
            diff.append(left > right)
    
    decimal = 0
    hex_str = []
    for idx, val in enumerate(diff):
        if val:
            decimal += 2 ** (idx % 8)
        if (idx % 8) == 7:
            hex_str.append(hex(decimal)[2:].rjust(2, '0'))
            decimal = 0
    return ''.join(hex_str)

def hamming_dist(h1, h2):
    return bin(int(h1, 16) ^ int(h2, 16)).count('1')

print("=" * 60)
print("Scanning dataset splits for Exact & Near-Duplicates...")
print("=" * 60)

records = []
valid_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

for split_name, folder in splits.items():
    if not folder.is_dir():
        print(f"[WARN] Folder not found: {folder}")
        continue
    for p in folder.rglob('*.*'):
        if p.suffix.lower() in valid_exts:
            try:
                data = p.read_bytes()
                md5 = hashlib.md5(data).hexdigest()
                with Image.open(p) as img:
                    phash = dhash(img)
                records.append({
                    'split': split_name,
                    'class': p.parent.name,
                    'path': p,
                    'rel': f"{split_name}/{p.parent.name}/{p.name}",
                    'md5': md5,
                    'phash': phash
                })
            except Exception as e:
                print(f"[WARN] Error reading {p}: {e}")

print(f"Total images scanned: {len(records)}")
split_counts = {}
for r in records:
    split_counts[r['split']] = split_counts.get(r['split'], 0) + 1
for s, c in split_counts.items():
    print(f"  - {s}: {c} images")

# 1. Exact Duplicates (MD5)
md5_map = {}
exact_dupes_within = []
exact_dupes_cross = []

for r in records:
    h = r['md5']
    if h in md5_map:
        prev = md5_map[h]
        pair = (prev, r)
        if prev['split'] == r['split']:
            exact_dupes_within.append(pair)
        else:
            exact_dupes_cross.append(pair)
    else:
        md5_map[h] = r

print("\n" + "-" * 60)
print(f"1. EXACT DUPLICATES (Identical file bytes / MD5):")
print(f"   - Across different splits (leakage): {len(exact_dupes_cross)}")
for a, b in exact_dupes_cross:
    print(f"     * {a['rel']} <==> {b['rel']}")
print(f"   - Within the same split: {len(exact_dupes_within)}")
for a, b in exact_dupes_within[:5]:
    print(f"     * {a['rel']} <==> {b['rel']}")
if len(exact_dupes_within) > 5:
    print(f"     ... and {len(exact_dupes_within) - 5} more within same split.")

# 2. Near-Duplicates (Perceptual dHash Hamming Distance <= 4)
# (Out of 64 bits, <= 4 bits difference indicates visually near-identical images)
near_dupes_cross = []
near_dupes_within = []

# Exclude pairs that are already exact byte duplicates
exact_pair_keys = set()
for a, b in exact_dupes_cross + exact_dupes_within:
    exact_pair_keys.add((a['rel'], b['rel']))
    exact_pair_keys.add((b['rel'], a['rel']))

n = len(records)
for i in range(n):
    for j in range(i + 1, n):
        r1, r2 = records[i], records[j]
        if (r1['rel'], r2['rel']) in exact_pair_keys:
            continue
        dist = hamming_dist(r1['phash'], r2['phash'])
        if dist <= 4:
            pair = (r1, r2, dist)
            if r1['split'] == r2['split']:
                near_dupes_within.append(pair)
            else:
                near_dupes_cross.append(pair)

print("\n" + "-" * 60)
print(f"2. NEAR-DUPLICATES (Perceptual dHash distance <= 4 of 64 bits):")
print(f"   - Across different splits (leakage): {len(near_dupes_cross)}")
for a, b, dist in near_dupes_cross[:15]:
    print(f"     * [dist={dist}] {a['rel']} <==> {b['rel']}")
if len(near_dupes_cross) > 15:
    print(f"     ... and {len(near_dupes_cross) - 15} more cross-split near-duplicates.")

print(f"   - Within the same split: {len(near_dupes_within)}")
for a, b, dist in near_dupes_within[:5]:
    print(f"     * [dist={dist}] {a['rel']} <==> {b['rel']}")
if len(near_dupes_within) > 5:
    print(f"     ... and {len(near_dupes_within) - 5} more within-split near-duplicates.")

print("\n" + "=" * 60)
print("SUMMARY FOR PAPER / THESIS:")
print(f"Total scanned: {len(records)} images ({split_counts.get('train', 0)} train, {split_counts.get('val', 0)} val, {split_counts.get('test', 0)} test)")
print(f"Exact cross-split duplicate pairs: {len(exact_dupes_cross)}")
print(f"Near cross-split duplicate pairs:  {len(near_dupes_cross)}")
print("=" * 60)
