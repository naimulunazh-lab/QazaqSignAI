/*
  Holistic landmark provider: face, head, body and both hands in one frame.
  The features retain each hand's position relative to the face, so a hand at
  the cheek and the same hand in open air are different inputs to the model.
*/
const FACE=[1,10,13,14,33,61,70,107,152,263,291,300,336];
const POSE=[0,7,8,9,10,11,12];
const POINTS=[...FACE.map(i=>['faceLandmarks',i]),...POSE.map(i=>['poseLandmarks',i])];
export const FEATURE_SIZE=(POINTS.length+42)*3+3; // face/head + pose + 2 hands + presence flags

export class GestureRecognizer {
  constructor({video,canvas,onFrame,onStatus}){this.video=video;this.canvas=canvas;this.ctx=canvas.getContext('2d');this.onFrame=onFrame;this.onStatus=onStatus;}
  async start(){
    if(!window.Holistic||!window.Camera)throw new Error('MediaPipe Holistic жүктелмеді. Бетті тану модулін тексеріңіз.');
    this.holistic=new window.Holistic({locateFile:file=>`/vendor/mediapipe/holistic/${file}`});
    this.holistic.setOptions({modelComplexity:1,smoothLandmarks:true,enableSegmentation:false,refineFaceLandmarks:true,minDetectionConfidence:.6,minTrackingConfidence:.6});
    this.holistic.onResults(results=>this.drawAndSend(results));
    this.camera=new window.Camera(this.video,{onFrame:async()=>this.holistic.send({image:this.video}),width:1280,height:720,facingMode:'user'});
    this.camera.start();this.onStatus('live');
  }
  drawAndSend(results){
    const w=this.canvas.clientWidth,h=this.canvas.clientHeight;if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}this.ctx.clearRect(0,0,w,h);
    const handStyle={color:'#d8f262',lineWidth:3};
    for(const hand of [results.leftHandLandmarks,results.rightHandLandmarks])if(hand?.length){window.drawConnectors(this.ctx,hand,window.HAND_CONNECTIONS,handStyle);window.drawLandmarks(this.ctx,hand,{color:'#f27d5e',lineWidth:1,radius:3});}
    if(results.poseLandmarks?.length){window.drawConnectors(this.ctx,results.poseLandmarks,window.POSE_CONNECTIONS,{color:'#b7d9cf',lineWidth:2});window.drawLandmarks(this.ctx,results.poseLandmarks,{color:'#fff4d5',lineWidth:1,radius:2});}
    if(results.faceLandmarks?.length){const marks=FACE.map(i=>results.faceLandmarks[i]);window.drawLandmarks(this.ctx,marks,{color:'#f7d47c',lineWidth:1,radius:2});}
    this.onFrame(results);
  }
  stop(){this.camera?.stop?.();this.holistic?.close?.();this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height);this.onStatus('idle');}
}

function vector(point,origin,scale){return point?[((point.x-origin.x)/scale),((point.y-origin.y)/scale),((point.z-origin.z)/scale)]:[0,0,0];}
function addPoints(out,points,indices,origin,scale){indices.forEach(index=>out.push(...vector(points?.[index],origin,scale)));}
export function extractFeatures(results){
  const face=results.faceLandmarks||[],pose=results.poseLandmarks||[];
  const nose=face[1]||pose[0]||{x:.5,y:.5,z:0};
  const leftShoulder=pose[11],rightShoulder=pose[12];
  const scale=Math.max(.05,leftShoulder&&rightShoulder?Math.hypot(leftShoulder.x-rightShoulder.x,leftShoulder.y-rightShoulder.y):.28);
  const out=[];addPoints(out,face,FACE,nose,scale);addPoints(out,pose,POSE,nose,scale);
  for(const hand of [results.leftHandLandmarks,results.rightHandLandmarks])for(let i=0;i<21;i++)out.push(...vector(hand?.[i],nose,scale));
  out.push(face.length?1:0,results.leftHandLandmarks?.length?1:0,results.rightHandLandmarks?.length?1:0);
  return out;
}

export class MotionBuffer {
  constructor(maxFrames=48){this.maxFrames=maxFrames;this.frames=[];}
  add(results){this.frames.push(extractFeatures(results));if(this.frames.length>this.maxFrames)this.frames.shift();}
  latest(){return this.frames;}
  clear(){this.frames=[];}
}

export class MockModelAdapter {
  predict(frames){
    if(!frames.length)return {score:0,message:'Кадрда бет немесе қол көрінбейді.'};
    const last=frames.at(-1),hasFace=last.at(-3)>0,hasHand=last.at(-2)>0||last.at(-1)>0;
    if(!hasHand)return {score:0,message:'Кемінде бір қолыңызды камераға толық көрсетіңіз.'};
    return {score:hasFace?72:58,message:hasFace?'Бет, бас және қол координаттары жиналып жатыр. Нақты баға үшін осы жест бойынша модельді оқытыңыз.':'Бетті де кадрға қосыңыз: ол ымның мағынасын айқындауы мүмкін.'};
  }
}

export class TemporalModelAdapter {
  constructor(model,labels,meta){this.model=model;this.labels=labels;this.meta=meta;}
  static async load(base='/models/gesture-classifier'){
    if(!window.tf)throw new Error('TensorFlow.js runtime is unavailable.');
    const [model,labels,meta]=await Promise.all([window.tf.loadLayersModel(`${base}/model.json`),fetch(`${base}/labels.json`).then(r=>r.ok?r.json():Promise.reject()),fetch(`${base}/model-meta.json`).then(r=>r.ok?r.json():Promise.reject())]);
    if(meta.featureSize!==FEATURE_SIZE)throw new Error('Модельдің landmark сұлбасы ескірген.');return new TemporalModelAdapter(model,labels,meta);
  }
  predict(frames,target){
    const {sequenceLength,featureSize}=this.meta;if(frames.length<sequenceLength)return {score:0,message:`Қимылды тағы ${sequenceLength-frames.length} кадр ұстап тұрыңыз.`};
    const sequence=frames.slice(-sequenceLength);const tensor=window.tf.tensor3d([sequence],[1,sequenceLength,featureSize]);const prediction=this.model.predict(tensor);const scores=Array.from(prediction.dataSync());tensor.dispose();prediction.dispose();
    const best=scores.indexOf(Math.max(...scores)),label=this.labels[best],confidence=Math.round(scores[best]*100),matches=label===target.id;
    return {label,score:matches?confidence:Math.max(0,100-confidence),message:matches&&confidence>=78?'Тамаша! Қимыл, орналасу және қозғалыс үлгісі танылды.':matches?'Жақсы, қимылды сәл анық әрі баяу орындаңыз.':`Қазір «${label||'басқа'}» жесті байқалды. Қолдың бағыты мен қозғалысын тексеріңіз.`};
  }
}
