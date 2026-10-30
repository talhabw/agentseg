"""Create deterministic GT-derived stress sequences for SAM3 tracking evaluation."""
from __future__ import annotations
import json, math, random, re
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'label_Data/torpedo_image_label'; OUT=ROOT/'label_Data/stress'; SEED=20261030

def natural(p):
 m=re.search(r'_(\d+)_',p.stem); return int(m.group(1)) if m else p.name

def load():
 ims=sorted((SRC/'images').glob('*.jpg'),key=natural); rows=[]
 for p in ims:
  ls=SRC/'labels'/(p.stem+'.txt'); boxes=[]
  for line in ls.read_text().splitlines():
   q=line.split();
   if len(q)>=5 and q[0]=='4': boxes.append(list(map(float,q[:5])))
  rows.append((p,boxes))
 return rows

def xyxy(b,w,h):
 _,x,y,bw,bh=b; return [int((x-bw/2)*w),int((y-bh/2)*h),int((x+bw/2)*w),int((y+bh/2)*h)]
def yolo(box,w,h):
 x1,y1,x2,y2=box; return [4,(x1+x2)/2/w,(y1+y2)/2/h,max(0,x2-x1)/w,max(0,y2-y1)/h]
def write_seq(name, indices, transforms, rows):
 d=OUT/name; (d/'images').mkdir(parents=True,exist_ok=True); (d/'labels').mkdir(exist_ok=True)
 meta={'name':name,'seed':SEED,'source_dir':str(SRC),'frames':[],'transform':transforms}
 empty=0
 for j,i in enumerate(indices):
  im,bs=rows[i]; img=cv2.imread(str(im)); h,w=img.shape[:2]; out=img; outbs=[]
  t=transforms.get('type')
  if t=='crop':
   cw,ch=transforms['crop_w'],transforms['crop_h']; cx,cy=transforms['centers'][j]; x0=max(0,min(w-cw,int(cx-cw/2))); y0=max(0,min(h-ch,int(cy-ch/2))); out=img[y0:y0+ch,x0:x0+cw]
   for b in bs:
    x1,y1,x2,y2=xyxy(b,w,h); x1,y1,x2,y2=max(x1,x0)-x0,max(y1,y0)-y0,min(x2,x0+cw)-x0,min(y2,y0+ch)-y0
    orig=max(1,(xyxy(b,w,h)[2]-xyxy(b,w,h)[0])*(xyxy(b,w,h)[3]-xyxy(b,w,h)[1])); vis=max(0,x2-x1)*max(0,y2-y1)
    if vis/orig>=.15: outbs.append(yolo([x1,y1,x2,y2],cw,ch))
  elif t=='occlusion':
   out=img.copy(); b=bs[0] if bs else None
   if b and transforms['coverage'][j]==0: outbs=[b]
   elif b:
    x1,y1,x2,y2=xyxy(b,w,h); frac=transforms['coverage'][j]; pw=int((x2-x1)*math.sqrt(frac)); ph=int((y2-y1)*math.sqrt(frac)); x=max(x1,x2-pw); y=max(y1,y2-ph); rng=np.random.default_rng(SEED+j); out[y:y+ph,x:x+pw]=rng.integers(0,256,(ph,pw,3),dtype=np.uint8)
    if frac<.9: outbs=[b]
  else: outbs=bs
  if t!='crop' and t!='occlusion': outbs=[b for b in bs]
  (d/'images'/f'{j}.jpg').write_bytes(cv2.imencode('.jpg',out)[1].tobytes()); txt='\n'.join(' '.join(f'{v:.6f}' for v in yolo(xyxy(b,out.shape[1],out.shape[0]),out.shape[1],out.shape[0])) for b in outbs)
  (d/'labels'/f'{j}.txt').write_text(txt+'\n' if txt else ''); empty+=not outbs
  meta['frames'].append({'index':j,'source_frame':im.name,'source_index':i,'empty_label':not outbs})
 meta['empty_labels']=empty; meta['frame_count']=len(indices)
 if name=='mid': meta['prompt_frame']=len(indices)//2
 (d/'meta.json').write_text(json.dumps(meta,indent=2)+'\n'); make_sheet(d)
 return len(indices),empty

def make_sheet(d):
 files=sorted((d/'images').glob('*.jpg'),key=lambda p:int(p.stem)); ids=np.linspace(0,len(files)-1,min(24,len(files)),dtype=int); thumbs=[]
 for i in ids:
  im=cv2.imread(str(files[i])); h,w=im.shape[:2]; lab=(d/'labels'/f'{i}.txt').read_text().split()
  if len(lab)>=5:
   b=[float(x) for x in lab[:5]]; x1,y1,x2,y2=xyxy(b,w,h); cv2.rectangle(im,(x1,y1),(x2,y2),(0,255,0),max(2,w//400))
  im=cv2.resize(im,(240,round(im.shape[0]*240/im.shape[1]))); cv2.putText(im,str(i),(5,18),0,.55,(0,255,255),2); thumbs.append(im)
 cols=6; rows=math.ceil(len(thumbs)/cols); cellh=max(x.shape[0] for x in thumbs); sheet=np.zeros((rows*cellh,cols*240,3),np.uint8)
 for k,x in enumerate(thumbs): sheet[k//cols*cellh:k//cols*cellh+x.shape[0],k%cols*240:k%cols*240+240]=x
 cv2.imwrite(str(d/'gt_sheet.jpg'),sheet)

def main():
 rows=load(); n=len(rows); mid=n//2; h,w=cv2.imread(str(rows[0][0])).shape[:2]; stats={}
 stats['sub4']=write_seq('sub4',list(range(0,n,4)),{'type':'subsample','step':4},rows)
 crop_w, crop_h=int(w*.6),int(h*.6); centers=[]
 for j in range(n):
  phase=j/(n-1); shift=math.sin(phase*2*math.pi)*w*.55; centers.append([w/2+shift,h/2])
 stats['exit']=write_seq('exit',list(range(n)),{'type':'crop','crop_w':crop_w,'crop_h':crop_h,'centers':centers,'pan':'sinusoidal; object exits and returns'},rows)
 o0=200; cov=[.6+.4*(1-abs(j-o0-9.5)/9.5) if o0<=j<o0+20 else 0. for j in range(n)]
 stats['occl']=write_seq('occl',list(range(n)),{'type':'occlusion','span':[o0,o0+19],'coverage':cov,'policy':'drop label when coverage >= 90%; deterministic noise patch'},rows)
 stats['mid']=write_seq('mid',list(range(n)),{'type':'identity','prompt_frame':mid},rows)
 print(json.dumps(stats))
if __name__=='__main__': main()
