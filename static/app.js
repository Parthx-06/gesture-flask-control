async function postJSON(url, data={}) {
  const res = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  return res.json();
}
async function getJSON(url){ const res = await fetch(url); return res.json(); }

async function refreshStatus(){
  try{
    const s = await getJSON('/api/status');
    document.getElementById('runState').textContent = s.running ? 'running' : 'stopped';
    document.getElementById('gesture').textContent = s.gesture || 'None';
    document.getElementById('volume').textContent = s.volume ?? 0;
  }catch(e){}
}
function bindHome(){
  const start=document.getElementById('startBtn');
  const stop =document.getElementById('stopBtn');
  const save =document.getElementById('saveCfg');

  if(start) start.onclick=async()=>{ await postJSON('/api/start'); refreshStatus(); };
  if(stop)  stop.onclick =async()=>{ await postJSON('/api/stop');  refreshStatus(); };
  if(save){
    save.onclick=async()=>{
      const data={
        scroll_sensitivity: Number(document.getElementById('scroll').value),
        mouse_sensitivity:  Number(document.getElementById('mouse').value),
        cooldown:           Number(document.getElementById('cooldown').value),
      };
      await postJSON('/api/config', data);
    };
  }
  refreshStatus();
  setInterval(refreshStatus, 700);
}
document.addEventListener('DOMContentLoaded', bindHome);
