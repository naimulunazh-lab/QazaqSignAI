"""Train a face + head + hands temporal classifier and export it for the browser."""
from __future__ import annotations
import argparse,json,pathlib
import numpy as np
import tensorflow as tf
FEATURE_SIZE=189
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='../dataset/sequences.npz');parser.add_argument('--out',default='../models/gesture-classifier');parser.add_argument('--epochs',type=int,default=45);args=parser.parse_args()
 data=np.load(args.dataset);X=data['X'].astype('float32');raw=data['y'].astype(str);signers=data['signers'].astype(str)
 if X.ndim!=3 or X.shape[2]!=FEATURE_SIZE:raise ValueError(f'Expected [takes,frames,{FEATURE_SIZE}], got {X.shape}')
 labels=sorted(set(raw));
 if len(labels)<2:raise ValueError('At least two gesture classes are needed.')
 lookup={name:index for index,name in enumerate(labels)};y=np.array([lookup[name] for name in raw])
 # Evaluation must use unseen people, never random frames from the same signer.
 valid_signer=sorted(set(signers))[-1];train=signers!=valid_signer;valid=~train
 if not valid.any() or not train.any():raise ValueError('Record at least two signers for a valid split.')
 model=tf.keras.Sequential([tf.keras.layers.Input((X.shape[1],FEATURE_SIZE)),tf.keras.layers.Conv1D(128,5,padding='same',activation='relu'),tf.keras.layers.BatchNormalization(),tf.keras.layers.Conv1D(96,3,padding='same',activation='relu'),tf.keras.layers.Dropout(.25),tf.keras.layers.GlobalAveragePooling1D(),tf.keras.layers.Dense(96,activation='relu'),tf.keras.layers.Dropout(.25),tf.keras.layers.Dense(len(labels),activation='softmax')])
 model.compile(optimizer=tf.keras.optimizers.Adam(3e-4),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
 model.fit(X[train],y[train],validation_data=(X[valid],y[valid]),epochs=args.epochs,batch_size=16,callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True)],verbose=2)
 out=pathlib.Path(args.out);out.mkdir(parents=True,exist_ok=True);model.save(out/'classifier.keras');(out/'labels.json').write_text(json.dumps(labels,ensure_ascii=False),encoding='utf-8');(out/'model-meta.json').write_text(json.dumps({'sequenceLength':int(X.shape[1]),'featureSize':FEATURE_SIZE},indent=2),encoding='utf-8')
 import tensorflowjs as tfjs;tfjs.converters.save_keras_model(model,str(out));print(f'Exported {len(labels)} temporal classes to {out}')
if __name__=='__main__':main()
