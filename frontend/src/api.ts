export async function api<T>(path:string, options:RequestInit={}):Promise<T> {
  const response=await fetch('/api'+path,{...options,headers:{...(options.body && !(options.body instanceof FormData)?{'Content-Type':'application/json'}:{}),...options.headers}});
  if(!response.ok){
    const body=await response.json().catch(()=>({detail:response.statusText}));
    const detail=typeof body.detail==='string'?body.detail:JSON.stringify(body.detail);
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return response.json();
}
export const post=<T>(path:string, body?:unknown)=>api<T>(path,{method:'POST',body:body===undefined?undefined:JSON.stringify(body)});
export function timeLabel(seconds:number, decimal=false){
  const s=Math.max(0,seconds); const h=Math.floor(s/3600),m=Math.floor(s%3600/60),sec=Math.floor(s%60);
  const basic=[h,m,sec].map(v=>String(v).padStart(2,'0')).join(':');
  return decimal?`${basic}.${Math.floor((s%1)*10)}`:basic;
}
export function parseTime(value:string):number|null {
  const text=value.trim(); if(!text)return null;
  if(/^\d+(\.\d+)?$/.test(text))return Number(text);
  if(!/^\d{1,3}:\d{2}:\d{2}(\.\d+)?$/.test(text))return null;
  const [h,m,s]=text.split(':').map(Number);
  return m<60&&s<60?h*3600+m*60+s:null;
}
export const labelName=(label:string)=>({seizure:'Possible seizure',abnormal:'Abnormal activity',LPD:'Lateralized periodic discharges',GPD:'Generalized periodic discharges',LRDA:'Lateralized rhythmic delta activity',GRDA:'Generalized rhythmic delta activity'}[label]||label);
export const stageColors:Record<string,string>={Awake:'#bca0ed',N1:'#91b7e5',N2:'#7c96d5',N3:'#536dac',REM:'#67c9be',Unscored:'#303640',Uncertain:'#8b8b9d'};
