import {useEffect,useRef,useState} from 'react';
import {api,stageColors,timeLabel} from './api';
import type {Event,Recording,Results,ScoreData,Selection,SleepEpoch,WaveformData} from './types';

export function EEGChart({recording,start,span,gain,montage,selected,channels}:{recording:Recording;start:number;span:number;gain:number;montage:string;selected:Selection|null;channels:string[]}){
  const ref=useRef<HTMLCanvasElement>(null),container=useRef<HTMLDivElement>(null);
  const [data,setData]=useState<WaveformData|null>(null),[error,setError]=useState(''),[loading,setLoading]=useState(false),[size,setSize]=useState(800);
  useEffect(()=>{const el=container.current;if(!el)return;const observer=new ResizeObserver(entries=>setSize(entries[0].contentRect.width));observer.observe(el);return()=>observer.disconnect()},[]);
  useEffect(()=>{const controller=new AbortController();setError('');setLoading(true);
    const params=new URLSearchParams({start:String(start),end:String(Math.min(recording.duration_s,start+span)),points:'1600',montage,channels:channels.join(',')});
    api<WaveformData>(`/recordings/${recording.id}/waveform?${params}`,{signal:controller.signal}).then(setData).catch(e=>{if(e.name!=='AbortError')setError(e.message)}).finally(()=>{if(!controller.signal.aborted)setLoading(false)});
    return()=>controller.abort();
  },[recording.id,start,span,montage,channels.join(',')]);
  useEffect(()=>{
    const canvas=ref.current;if(!canvas||!data)return;const dpr=window.devicePixelRatio||1;
    const width=size,height=Math.max(300,data.channels.length*34+28);canvas.width=width*dpr;canvas.height=height*dpr;canvas.style.height=`${height}px`;
    const ctx=canvas.getContext('2d')!;ctx.scale(dpr,dpr);ctx.fillStyle='#141922';ctx.fillRect(0,0,width,height);
    const left=62,right=width-18,plot=right-left,x=(t:number)=>left+(t-start)/span*plot;
    if(selected){const a=Math.max(start,selected.item.start_s),b=Math.min(start+span,selected.item.end_s);if(b>a){ctx.fillStyle=selected.type==='sleep'?'#70a1e51a':'#ee9a6714';ctx.fillRect(x(a),0,x(b)-x(a),height-25)}}
    ctx.font='10px ui-monospace, SFMono-Regular, monospace';
    const step=span<=15?1:span<=40?5:10;
    for(let t=Math.ceil(start/step)*step;t<=start+span;t+=step){ctx.strokeStyle='#28303a';ctx.beginPath();ctx.moveTo(x(t),0);ctx.lineTo(x(t),height-26);ctx.stroke();ctx.fillStyle='#738094';ctx.fillText(timeLabel(t),x(t)-22,height-8)}
    data.channels.forEach((channel,i)=>{
      const base=18+i*34;ctx.fillStyle='#8590a3';ctx.fillText(channel.name.replace(/^EEG\s*/i,''),7,base+3);ctx.strokeStyle='#212833';ctx.beginPath();ctx.moveTo(left,base);ctx.lineTo(right,base);ctx.stroke();
      ctx.save();ctx.beginPath();ctx.rect(left,base-17,plot,34);ctx.clip();ctx.strokeStyle=i%2===0?'#80c5bd':'#8fbcb8';ctx.lineWidth=.75;ctx.beginPath();let begun=false;
      data.times.forEach((t,j)=>{const lo=channel.min[j],hi=channel.max[j];if(lo===null||hi===null){begun=false;return;}const px=x(t),y1=base-lo/gain*14,y2=base-hi/gain*14;if(!begun){ctx.moveTo(px,y1);begun=true;}else ctx.lineTo(px,y1);ctx.lineTo(px,y2)});ctx.stroke();ctx.restore();
    });
  },[data,size,start,span,gain,selected]);
  return <div className="wave-canvas" ref={container} aria-busy={loading}>
    {error?<div className="chart-message error">{error}</div>:<canvas ref={ref} role="img" aria-label={`EEG waveform, ${channels.length} channels from ${timeLabel(start)} to ${timeLabel(start+span)}, ${gain} microvolt scale. ${recording.kind==='synthetic'?'Synthetic demonstration.':''}`}/>}
    {loading&&<span className="chart-loading">Loading signal…</span>}
  </div>
}

export function Overview({recording,results,start,span,onSeek}:{recording:Recording;results:Results|null;start:number;span:number;onSeek:(n:number)=>void}){
  const duration=recording.duration_s;
  return <div className="overview">
    <div className="overview-lanes" onClick={e=>{const rect=e.currentTarget.getBoundingClientRect();onSeek((e.clientX-rect.left)/rect.width*duration)}} aria-label="Recording overview">
      <span className="lane-name">Events</span>
      {results?.events.map(e=><span title={`${labelNameShort(e.label)} · ${timeLabel(e.start_s)}`} key={e.id} className={`overview-event ${e.label==='seizure'?'seizure':'abnormal'}`} style={{left:`${e.start_s/duration*100}%`,width:`${Math.max(.3,(e.end_s-e.start_s)/duration*100)}%`}}/>)}
      <div className="overview-sleep">{results?.sleep.map(e=><span key={e.id} title={e.stage} style={{left:`${e.start_s/duration*100}%`,width:`${(e.end_s-e.start_s)/duration*100}%`,background:stageColors[e.stage]}}/>)}</div>
      <span className="viewport-marker" style={{left:`${start/duration*100}%`,width:`${Math.max(.7,span/duration*100)}%`}}/>
    </div>
    <div className="overview-ticks">{[0,.25,.5,.75,1].map(p=><span key={p}>{timeLabel(p*duration)}</span>)}</div>
  </div>
}
const labelNameShort=(s:string)=>s==='seizure'?'Possible seizure':s;

export function SleepChart({epochs,duration,onSelect,selectedId}:{epochs:SleepEpoch[];duration:number;onSelect:(e:SleepEpoch)=>void;selectedId?:string}){
  const stages=['Awake','REM','N1','N2','N3'];
  const y=(stage:string)=>20+stages.indexOf(stage)*27;
  const path=epochs.filter(e=>stages.includes(e.stage)).map(e=>`M ${e.start_s/duration*940+55} ${y(e.stage)} H ${e.end_s/duration*940+55}`).join(' ');
  return <div className="hypnogram">
    <svg viewBox="0 0 1010 154" role="img" aria-label="Predicted sleep stages over the recording">
      {stages.map(stage=><g key={stage}><text x="3" y={y(stage)+4}>{stage}</text><line x1="55" x2="995" y1={y(stage)} y2={y(stage)} stroke="#29313c"/></g>)}
      {epochs.map(e=><rect key={e.id} x={55+e.start_s/duration*940} y="6" width={Math.max(.8,(e.end_s-e.start_s)/duration*940)} height="132" fill={stageColors[e.stage]} opacity={e.id===selectedId?0.45:stages.includes(e.stage)?0.05:0.4} onClick={()=>onSelect(e)}><title>{`${timeLabel(e.start_s)} · ${e.stage}`}</title></rect>)}
      <path d={path} fill="none" stroke="#9baded" strokeWidth="3" pointerEvents="none"/>
    </svg>
    <div className="sleep-legend">{[...stages,'Unscored'].map(s=><span key={s}><i style={{background:stageColors[s]}}/>{s}</span>)}</div>
    <label className="sr-only" htmlFor="sleep-epoch">Select sleep epoch</label><select id="sleep-epoch" className="sleep-epoch-select" value={selectedId||''} onChange={e=>{const epoch=epochs.find(v=>v.id===e.target.value);if(epoch)onSelect(epoch)}}><option value="">Inspect a sleep epoch…</option>{epochs.map(e=><option key={e.id} value={e.id}>{timeLabel(e.start_s)} — {e.stage}</option>)}</select>
  </div>
}

export function ScoreTrack({runId,start,span}:{runId:string;start:number;span:number}){
  const [data,setData]=useState<ScoreData|null>(null);
  useEffect(()=>{const controller=new AbortController();api<ScoreData>(`/analyses/${runId}/scores?head=seizure&start=${start}&end=${start+span}`,{signal:controller.signal}).then(setData).catch(()=>setData(null));return()=>controller.abort()},[runId,start,span]);
  if(!data?.times.length)return <div className="score-empty">Seizure scores are not available for this interval.</div>;
  let path='',pen=false;
  data.times.forEach((t,i)=>{const score=data.scores[i][1];if(!data.valid[i]||score===null){pen=false;return;}path+=`${pen?'L':'M'} ${(t-start)/span*1000} ${42-score*35} `;pen=true});
  return <div className="score-track"><div><span>Seizure model score</span><span>0—1 · not a clinical probability</span></div><svg viewBox="0 0 1000 48" preserveAspectRatio="none" role="img" aria-label="Seizure model score across the displayed interval"><line x1="0" y1="42" x2="1000" y2="42" stroke="#303747"/><path d={path} fill="none" stroke="#ecaa86" strokeWidth="2"/></svg></div>
}
