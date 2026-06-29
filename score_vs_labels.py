import csv, sys, math, statistics as st
TOL = 60.0
def load_labels(p):
    L={}
    for r in csv.DictReader(open(p)):
        u=r['u']; v=r['v']
        L[int(r['frame'])]=(r['visible']=='1', float(u) if u not in ('','-1') else None,
                            float(v) if v not in ('','-1') else None)
    return L
def score(run, L):
    locs={}
    for r in csv.DictReader(open(f'{run}/ball_dual_locations.csv')):
        u=r['cam1_x_px']; v=r['cam1_y_px']
        locs[int(r['frame'])]=(r['cam1_detected']=='1', float(u) if u else None, float(v) if v else None)
    TP=FP=FN=TN=0; errs=[]
    for f,(vis,lu,lv) in L.items():
        if f not in locs: continue
        det,u,v = locs[f]; present = det and u is not None
        if vis and lu is not None:
            if present and math.hypot(u-lu,v-lv)<=TOL: TP+=1; errs.append(math.hypot(u-lu,v-lv))
            elif present: FP+=1; FN+=1
            else: FN+=1
        else:
            FP += 1 if present else 0; TN += 0 if present else 1
    prec=TP/(TP+FP) if TP+FP else 0; rec=TP/(TP+FN) if TP+FN else 0
    med=st.median(errs) if errs else float('nan')
    print(f'{run}: precision={prec:.3f} recall={rec:.3f} med_err={med:.1f}px (TP={TP} FP={FP} FN={FN})')
L=load_labels('output/ball_labels_side-1-full-vid.csv')
for run in sys.argv[1:]: score(run, L)
