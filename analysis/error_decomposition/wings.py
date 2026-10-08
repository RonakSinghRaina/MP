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
rows=[]
for i in range(109):
    img=X[i].astype(np.float64); y=Y[i]
    if not y.any(): continue
    clean=img[~y]; med=np.median(clean); mad=1.4826*np.median(np.abs(clean-med))+1e-9
    snr=(img-med)/mad                                  # brightness in units of the clean-pixel scatter
    lab,n=ndi.label(y,structure=np.ones((3,3)))
    for j,sl in enumerate(ndi.find_objects(lab),start=1):
        obj=lab[sl]==j
        peak=snr[sl][obj].max()
        wf=obj.sum(axis=1).max()                       # widest extent along FREQUENCY (cols) in any row
        wt=obj.sum(axis=0).max()                       # widest extent along TIME (rows)
        pobj=pred[i][sl][obj]
        rows.append((peak,wf,wt,obj.sum(),pobj.mean()))
R=np.array(rows)
bins=[(-1e9,3),(3,10),(10,30),(30,100),(100,1e9)]
print(f"{'peak brightness (clean sigma)':<30}{'objects':>8}{'median freq-width':>19}{'median time-width':>19}{'mean px/object':>16}{'model recall in obj':>21}")
for a,b in bins:
    s=(R[:,0]>=a)&(R[:,0]<b)
    if s.sum()==0: continue
    lab=f"{a if a>-1e8 else '<'}–{b if b<1e8 else '+'}"
    print(f"{lab:<30}{s.sum():>8}{np.median(R[s,1]):>19.1f}{np.median(R[s,2]):>19.1f}{R[s,3].mean():>16.1f}{R[s,4].mean():>21.3f}")
from scipy.stats import spearmanr
print("\nSpearman correlation, peak brightness vs freq-width: %.3f   vs time-width: %.3f" % (spearmanr(R[:,0],R[:,1])[0], spearmanr(R[:,0],R[:,2])[0]))
