"""Convert JSON takes downloaded from capture.html into sequences.npz."""
from __future__ import annotations
import argparse,json,pathlib
import numpy as np
FEATURE_SIZE=189
def resample(frames,length):
    indexes=np.linspace(0,len(frames)-1,length).round().astype(int)
    return np.asarray(frames,dtype=np.float32)[indexes]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',default='../dataset/recordings');parser.add_argument('--out',default='../dataset/sequences.npz');parser.add_argument('--frames',type=int,default=48);args=parser.parse_args()
    samples=[];labels=[];signers=[]
    for path in pathlib.Path(args.input).glob('*.json'):
        data=json.loads(path.read_text(encoding='utf-8'));frames=data.get('frames',[])
        if data.get('feature_size')!=FEATURE_SIZE or len(frames)<8: print(f'Skipped {path.name}');continue
        array=resample(frames,args.frames)
        if array.shape!=(args.frames,FEATURE_SIZE): print(f'Skipped malformed {path.name}');continue
        samples.append(array);labels.append(data['gesture_id']);signers.append(data.get('signer_id','unknown'))
    if not samples: raise SystemExit('No valid recordings found.')
    np.savez_compressed(args.out,X=np.stack(samples),y=np.asarray(labels),signers=np.asarray(signers));print(f'Wrote {len(samples)} takes to {args.out}')
if __name__=='__main__':main()
