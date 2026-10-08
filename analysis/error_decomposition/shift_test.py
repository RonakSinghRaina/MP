import json, os, sys, hashlib, numpy as np, torch
from scipy import ndimage as ndi
ROOT=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"analysis"))
from diagnose_context_headroom import predict, RUN, HybridRFINet
from lofar_data import load_lofar
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
m=json.load(open(os.path.join(RUN,"eval_test","metrics.json"))); lo,hi=m["fixed_range"]; th=float(m["val_selected_threshold"])
model=HybridRFINet(1,2,base=m["base"],depth=m["depth"],dropout=m["dropout"])
ck=torch.load(os.path.join(RUN,"best.pt"),map_location="cpu",weights_only=False); model.load_state_dict(ck.get("model",ck)); model.to(dev).eval()
d=load_lofar(); Y=np.asarray(d.test_masks[...,0],bool)
P=predict(model,[d.test_images[i,:,:,0] for i in range(109)],lo,hi,dev); pred=P>th
leak=np.load(os.path.join(ROOT,"analysis","lofar_analysis","lofar_leak_train_idx.npy"))
h=lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
tr_of={h(d.train_images[int(i)]):int(i) for i in leak}
A=np.stack([np.asarray(d.train_masks[tr_of[h(d.test_images[t])],:,:,0],bool) for t in range(109)])
def f1(M,T):
    tp=(M&T).sum(); fp=(M&~T).sum(); fn=(~M&T).sum(); p=tp/max(tp+fp,1); r=tp/max(tp+fn,1); return 2*p*r/max(p+r,1e-12)
def shift(M,dy,dx):
    return np.roll(np.roll(M,dy,axis=1),dx,axis=2)
print("F1 vs EXPERT after shifting the mask by (rows=time dy, cols=freq dx):")
print("   dy dx   AOFlagger   model")
for dy in (-2,-1,0,1,2):
    for dx in (-2,-1,0,1,2):
        if abs(dy)+abs(dx)>2: continue
        print(f"   {dy:>2} {dx:>2}   {f1(shift(A,dy,dx),Y):.4f}     {f1(shift(pred,dy,dx),Y):.4f}")
print("\nMorphology on the MODEL mask (diagnostic, tuned on test):")
st={"dilate time(rows) 1":np.array([[1],[1],[1]]),"dilate freq(cols) 1":np.array([[1,1,1]]),"dilate 3x3":np.ones((3,3))}
for k,s in st.items():
    D=np.stack([ndi.binary_dilation(p,structure=s) for p in pred]); print(f"   {k:<22} {f1(D,Y):.4f}")
    E=np.stack([ndi.binary_erosion(p,structure=s) for p in pred]); print(f"   {k.replace('dilate','erode'):<22} {f1(E,Y):.4f}")
print("\nSame for AOFlagger mask vs expert:")
for k,s in st.items():
    D=np.stack([ndi.binary_dilation(p,structure=s) for p in A]); print(f"   {k:<22} {f1(D,Y):.4f}")
    E=np.stack([ndi.binary_erosion(p,structure=s) for p in A]); print(f"   {k.replace('dilate','erode'):<22} {f1(E,Y):.4f}")
# AOFlagger boundary share
FN=Y&~A; FP=A&~Y
dA=np.stack([ndi.distance_transform_edt(~a) if a.any() else np.full(a.shape,999.) for a in A])
dY=np.stack([ndi.distance_transform_edt(~y) if y.any() else np.full(y.shape,999.) for y in Y])
print(f"\nAOFlagger vs expert: missed within 2px of an AOFlagger flag {100*(FN&(dA<=2)).sum()/FN.sum():.1f}%, "
      f"false alarms within 2px of expert RFI {100*(FP&(dY<=2)).sum()/FP.sum():.1f}%")
# mask widths: mean run length along freq (cols) and time (rows) for expert / AOFlagger / model
def runlen(M,axis):
    B=M if axis==2 else M.transpose(0,2,1)
    pad=np.zeros(B.shape[:2]+(1,),bool); Z=np.concatenate([pad,B,pad],axis=2)
    starts=(~Z[:,:,:-1]&Z[:,:,1:]).sum(); return B.sum()/max(starts,1)
for name,M in (("expert",Y),("AOFlagger",A),("model",pred)):
    print(f"   {name:<10} mean run length along freq {runlen(M,2):.2f} px, along time {runlen(M,1):.2f} px, RFI frac {100*M.mean():.3f}%")
