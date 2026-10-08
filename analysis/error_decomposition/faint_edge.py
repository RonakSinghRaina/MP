import json, os, sys, numpy as np, torch
from scipy import ndimage as ndi
ROOT=os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,"analysis"))
from diagnose_context_headroom import predict, RUN, HybridRFINet
from ablation_faint_recall import normalise_fixed
from lofar_data import load_lofar
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
m=json.load(open(os.path.join(RUN,"eval_test","metrics.json"))); lo,hi=m["fixed_range"]; th=float(m["val_selected_threshold"])
model=HybridRFINet(1,2,base=m["base"],depth=m["depth"],dropout=m["dropout"])
ck=torch.load(os.path.join(RUN,"best.pt"),map_location="cpu",weights_only=False); model.load_state_dict(ck.get("model",ck)); model.to(dev).eval()
d=load_lofar(); Y=np.asarray(d.test_masks[...,0],bool)
X=[d.test_images[i,:,:,0] for i in range(109)]
P=predict(model,X,lo,hi,dev); pred=P>th
tier=np.full(Y.shape,-1,np.int8)
for i in range(109):
    img=normalise_fixed(X[i],lo,hi); msk=Y[i]
    if not msk.any(): continue
    clean=img[~msk]; c50,c75,c99=np.percentile(clean,[50,75,99]); v=img[msk]
    t=np.full(v.shape,3,np.int8); t[v>=c50]=2; t[v>=c75]=1; t[v>=c99]=0; tier[i][msk]=t
dist_pred=np.stack([ndi.distance_transform_edt(~p) if p.any() else np.full(p.shape,999.) for p in pred])
# distance of each true-RFI pixel to the nearest NON-RFI pixel (1 = on the object's edge)
depth=np.stack([ndi.distance_transform_edt(y) for y in Y])
FN=Y&~pred
names=["bright","moderate","faint","invisible"]
print(f"{'tier':<10}{'RFI px':>9}{'missed':>9}{'recall':>8}{'missed<=1px of detection':>27}{'on object edge (depth 1)':>26}")
for k in range(4):
    T=tier==k; miss=FN&T
    print(f"{names[k]:<10}{T.sum():>9,}{miss.sum():>9,}{(pred&T).sum()/T.sum():>8.3f}"
          f"{100*(miss&(dist_pred<=1)).sum()/max(miss.sum(),1):>26.1f}%{100*(T&(depth<=1)).sum()/T.sum():>25.1f}%")
print(f"\nall RFI pixels on an object edge (depth 1): {100*(Y&(depth<=1)).sum()/Y.sum():.1f}%")
