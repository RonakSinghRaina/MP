import pickle, hashlib
import numpy as np

with open('/home/ronaksingh/Documents/minor project/Minor Project/LOFAR_Full_RFI_dataset.pkl','rb') as f:
    data = pickle.load(f)
TRX, TRY, TEX, TEY = data

def md5(a): return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()

# ---- find, for each test image, a train image that is byte-identical
tr_hash = {}
for i in range(TRX.shape[0]):
    tr_hash.setdefault(md5(TRX[i]), i)

print("=" * 78)
print("TEST 1 -- is every test image found somewhere in train?")
print("=" * 78)
pairs = []
for t in range(TEX.shape[0]):
    h = md5(TEX[t])
    pairs.append((t, tr_hash.get(h, None)))
found = [p for p in pairs if p[1] is not None]
print(f"test images: {TEX.shape[0]}")
print(f"test images that have a byte-identical twin in train: {len(found)}")
print()

print("=" * 78)
print("TEST 2 -- element-by-element equality, not just hashes")
print("=" * 78)
n_exact = 0
worst = 0.0
for t, i in found:
    same = np.array_equal(TEX[t], TRX[i])
    d = float(np.abs(TEX[t].astype(np.float64) - TRX[i].astype(np.float64)).max())
    worst = max(worst, d)
    n_exact += int(same)
print(f"pairs where np.array_equal(...) is True : {n_exact} / {len(found)}")
print(f"largest pixel difference across ALL pairs: {worst}")
print()

print("=" * 78)
print("TEST 3 -- show five actual pairs")
print("=" * 78)
print(f"{'test idx':>8} {'train idx':>10}  {'md5 of test image':>34}  equal?")
for t, i in found[:5]:
    print(f"{t:>8} {i:>10}  {md5(TEX[t])}  {np.array_equal(TEX[t], TRX[i])}")
print()

print("=" * 78)
print("TEST 4 -- CONTROL: are all images just identical to each other?")
print("=" * 78)
t0, i0 = found[0]
others = [j for j in range(0, 2000, 331) if j != i0][:5]
print(f"test[{t0}] vs its matched train[{i0}] : equal = {np.array_equal(TEX[t0], TRX[i0])}")
for j in others:
    print(f"test[{t0}] vs UNRELATED  train[{j:>4}] : equal = {np.array_equal(TEX[t0], TRX[j])}"
          f"   (max diff {np.abs(TEX[t0].astype(np.float64)-TRX[j].astype(np.float64)).max():.4g})")
print()

print("=" * 78)
print("TEST 5 -- the IMAGES match but the MASKS do not (different labellers)")
print("=" * 78)
print(f"{'test idx':>8} {'train idx':>10} {'image same':>11} {'mask same':>10} "
      f"{'AOFlag %':>9} {'human %':>8} {'agree %':>8}")
for t, i in found[:8]:
    a = TRY[i, :, :, 0]; h = TEY[t, :, :, 0]
    print(f"{t:>8} {i:>10} {str(np.array_equal(TEX[t],TRX[i])):>11} "
          f"{str(np.array_equal(a,h)):>10} {a.mean()*100:>8.3f}% {h.mean()*100:>7.3f}% "
          f"{(a==h).mean()*100:>7.3f}%")
n_mask_same = sum(np.array_equal(TRY[i,:,:,0], TEY[t,:,:,0]) for t, i in found)
print(f"\npairs where the MASKS are also identical: {n_mask_same} / {len(found)}")
