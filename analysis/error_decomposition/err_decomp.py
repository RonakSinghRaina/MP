import json, os, sys, numpy as np, torch
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
FN=Y&~pred; FP=pred&~Y; nfn=FN.sum(); nfp=FP.sum()
print(f"FN {nfn:,}  FP {nfp:,}")
# E1/E2: distance of each error pixel to the nearest predicted / true RFI pixel (per image)
dist_pred=np.stack([ndi.distance_transform_edt(~p) if p.any() else np.full(p.shape,999.) for p in pred])
dist_true=np.stack([ndi.distance_transform_edt(~y) if y.any() else np.full(y.shape,999.) for y in Y])
for k in (1,2,3,5,10):
    print(f"E1 missed pixels within {k:>2}px of a detection: {100*(FN&(dist_pred<=k)).sum()/nfn:5.1f}%   |   "
          f"E2 false alarms within {k:>2}px of true RFI: {100*(FP&(dist_true<=k)).sum()/nfp:5.1f}%")
# E3: objects
def comps(M):
    out=[]
    for i in range(len(M)):
        lab,n=ndi.label(M[i],structure=np.ones((3,3)))
        out.append((lab,n))
    return out
cY=comps(Y); cP=comps(pred)
miss_obj=miss_px=tot_obj=0; sizes_missed=[]
for i,(lab,n) in enumerate(cY):
    if n==0: continue
    sz=np.bincount(lab.ravel())[1:]; hit=np.bincount(lab[pred[i]].ravel(),minlength=n+1)[1:]
    tot_obj+=n; mm=hit==0; miss_obj+=mm.sum(); miss_px+=sz[mm].sum(); sizes_missed+=list(sz[mm])
print(f"E3 true RFI objects: {tot_obj:,}; entirely missed: {miss_obj:,} ({100*miss_obj/tot_obj:.1f}%), "
      f"holding {miss_px:,} px = {100*miss_px/nfn:.1f}% of all missed pixels; median missed-object size {np.median(sizes_missed):.0f} px")
fo=fo_px=tp_obj=0
for i,(lab,n) in enumerate(cP):
    if n==0: continue
    sz=np.bincount(lab.ravel())[1:]; hit=np.bincount(lab[Y[i]].ravel(),minlength=n+1)[1:]
    tp_obj+=n; ff=hit==0; fo+=ff.sum(); fo_px+=sz[ff].sum()
print(f"E3 predicted objects: {tp_obj:,}; with NO overlap with truth (false objects): {fo:,} ({100*fo/tp_obj:.1f}%), "
      f"holding {fo_px:,} px = {100*fo_px/nfp:.1f}% of all false-alarm pixels")
# E4: false alarms inside channels the model flags >=80%
cf=pred.mean(axis=1,keepdims=True)
print(f"E4 false-alarm pixels inside channels the model flags >=80%: {100*(FP&(cf>=0.8)).sum()/nfp:.1f}%")
# what if we only fixed boundary errors (within 2px)? and only object-level errors?
def f1(tp,fp,fn): p=tp/(tp+fp); r=tp/(tp+fn); return 2*p*r/(p+r)
tp=(pred&Y).sum()
b_fn=(FN&(dist_pred<=2)).sum(); b_fp=(FP&(dist_true<=2)).sum()
print(f"IF boundary errors (<=2px) were fixed: F1 {f1(tp+b_fn, nfp-b_fp, nfn-b_fn):.4f}")
print(f"IF whole missed objects were found: F1 {f1(tp+miss_px, nfp, nfn-miss_px):.4f}")
print(f"IF false objects were removed:     F1 {f1(tp, nfp-fo_px, nfn):.4f}")
