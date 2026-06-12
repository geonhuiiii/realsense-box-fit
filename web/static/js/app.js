import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";

const EDGES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
const PALETTE = [0xe64d3d,0x2ecc71,0x3498db,0xf39c12,0x9b59b6,0x1abc9c,0xf1c40f,0xff7eb6,0x55efc4,0xa29bfe];

/* ---- a single 3D view (scene + camera + controls + label overlay) ---- */
function makeView(elId){
  const el = document.getElementById(elId);
  const renderer = new THREE.WebGLRenderer({antialias:true});
  renderer.setPixelRatio(devicePixelRatio);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.appendChild(renderer.domElement);

  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.style.position="absolute";
  labelRenderer.domElement.style.top="0";
  labelRenderer.domElement.style.left="0";
  labelRenderer.domElement.style.pointerEvents="none";
  el.appendChild(labelRenderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0c12);
  const cam = new THREE.PerspectiveCamera(50,1,0.1,100000);
  cam.up.set(0,0,1);
  cam.position.set(120,-160,120);
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.08;

  scene.add(new THREE.AmbientLight(0xffffff,0.7));
  const dir = new THREE.DirectionalLight(0xffffff,0.5); dir.position.set(1,-1,2); scene.add(dir);

  const grid = new THREE.GridHelper(400,40,0x3a4356,0x232838);
  grid.rotation.x = Math.PI/2; scene.add(grid);
  const axes = new THREE.AxesHelper(30); scene.add(axes);

  const content = new THREE.Group(); scene.add(content);
  const labels = new THREE.Group(); scene.add(labels);

  function resize(){
    const w = el.clientWidth, h = el.clientHeight;
    if(!w||!h) return;
    renderer.setSize(w,h); labelRenderer.setSize(w,h);
    cam.aspect = w/h; cam.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(el); resize();

  function fit(min,max){
    const c = new THREE.Vector3((min[0]+max[0])/2,(min[1]+max[1])/2,(min[2]+max[2])/2);
    const d = Math.max(10, new THREE.Vector3(max[0]-min[0],max[1]-min[1],max[2]-min[2]).length());
    controls.target.copy(c);
    cam.position.set(c.x + d*0.9, c.y - d*1.1, c.z + d*0.8);
    cam.near = d/500; cam.far = d*50; cam.updateProjectionMatrix();
  }
  function clear(g){ while(g.children.length) g.remove(g.children[0]); }

  return {el,renderer,labelRenderer,scene,cam,controls,grid,axes,content,labels,fit,
          clearAll(){clear(content);clear(labels);}};
}

const L = makeView("viewL");
const R = makeView("viewR");
const views = [L,R];

function animate(){
  requestAnimationFrame(animate);
  for(const v of views){ v.controls.update(); v.renderer.render(v.scene,v.cam);
    v.labelRenderer.render(v.scene,v.cam); }
}
animate();

/* ---- geometry helpers ---- */
function pointsObj(points, colors, size=1.3){
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(points.length*3), col = new Float32Array(points.length*3);
  for(let i=0;i<points.length;i++){
    pos[3*i]=points[i][0]; pos[3*i+1]=points[i][1]; pos[3*i+2]=points[i][2];
    const c = colors[i]||[0.6,0.6,0.6]; col[3*i]=c[0]; col[3*i+1]=c[1]; col[3*i+2]=c[2];
  }
  g.setAttribute("position",new THREE.BufferAttribute(pos,3));
  g.setAttribute("color",new THREE.BufferAttribute(col,3));
  return new THREE.Points(g, new THREE.PointsMaterial({size,vertexColors:true,sizeAttenuation:true}));
}
function edgesFromCorners(corners, color, opacity=0.95){
  const pos=[];
  for(const [a,b] of EDGES) pos.push(...corners[a],...corners[b]);
  const g=new THREE.BufferGeometry();
  g.setAttribute("position",new THREE.Float32BufferAttribute(pos,3));
  return new THREE.LineSegments(g,new THREE.LineBasicMaterial({color,transparent:true,opacity}));
}
function boxCorners(min,size){
  const [x,y,z]=min,[a,b,c]=size;
  return [[x,y,z],[x+a,y,z],[x+a,y+b,z],[x,y+b,z],
          [x,y,z+c],[x+a,y,z+c],[x+a,y+b,z+c],[x,y+b,z+c]];
}
function label(text,pos){
  const d=document.createElement("div"); d.className="label3d"; d.textContent=text;
  const o=new CSS2DObject(d); o.position.set(pos[0],pos[1],pos[2]); return o;
}

/* ---- render left: segmentation + OBB ---- */
function renderLeft(viz){
  L.clearAll();
  for(const o of viz.instances){
    const kept=o.kept!==false;
    const col = kept ? (PALETTE[o.id%PALETTE.length]) : 0x555a66;
    const p = pointsObj(o.points,o.colors); p.userData.kind="points";
    if(!kept) p.material.opacity=0.4, p.material.transparent=true;
    L.content.add(p);
    const e = edgesFromCorners(o.obb.corners,col,kept?0.95:0.4); e.userData.kind="obb";
    L.content.add(e);
    const cen=o.obb.corners.reduce((s,c)=>[s[0]+c[0]/8,s[1]+c[1]/8,s[2]+c[2]/8],[0,0,0]);
    L.labels.add(label(`${o.identity||o.label} ${o.dims_cm.join("×")}`,cen));
  }
  L.fit(viz.scene_bbox.min,viz.scene_bbox.max);
  applyToggles();
}

/* ---- render right: truck loading (boxes + loose furniture packed in truck) ---- */
const BOX_COL=0x4cc9f0, LOOSE_COL=0xf7b731, TRUCK_COL=0x7aed9b;
function renderRight(viz){
  R.clearAll();
  const tp=viz.truck_plan;
  if(!tp || tp.empty || tp.oversize || !tp.truck){
    R.fit([0,0,0],[150,150,150]); applyToggles(); return;
  }
  const [L,W,H]=tp.truck.dims_cm;
  let xoff=0, gap=80; const mx=[L,W,H];
  (tp.trucks||[]).forEach((load,ti)=>{
    // truck cargo wireframe
    R.content.add(edgesFromCorners(boxCorners([xoff,0,0],[L,W,H]),TRUCK_COL,0.95));
    R.labels.add(label(`${tp.truck.name} #${ti+1}`,[xoff+L/2,W/2,H+12]));
    for(const it of load.items){
      const s=it.size_cm;                       // [x,y,z], already truck frame
      const p=[xoff+it.pos_cm[0],it.pos_cm[1],it.pos_cm[2]];
      const col=it.kind==="loose"?LOOSE_COL:BOX_COL;
      const e=edgesFromCorners(boxCorners(p,s),col,0.9); e.userData.kind="obb";
      R.content.add(e);
      const fg=new THREE.BoxGeometry(s[0],s[1],s[2]);
      const fm=new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.18});
      const mesh=new THREE.Mesh(fg,fm);
      mesh.position.set(p[0]+s[0]/2,p[1]+s[1]/2,p[2]+s[2]/2);
      mesh.userData.kind="fill"; R.content.add(mesh);
    }
    mx[0]=Math.max(mx[0],xoff+L); xoff += L + gap;
  });
  R.fit([0,0,0],mx);
  applyToggles();
}

/* ---- table + summary ---- */
function renderTable(viz){
  const t=document.getElementById("obj-table");
  let h=`<tr><th></th><th>id</th><th>identity</th><th>material</th>
    <th>measured (cm)</th><th>corrected (cm)</th><th>box</th><th>flags</th><th>VLM note</th></tr>`;
  const boxOf={}, crammed=new Set();
  (viz.packing.boxes||[]).forEach(b=>b.items.forEach(it=>{
    boxOf[it.instance_id]=b.name; if(it.crammed) crammed.add(it.instance_id);}));
  const unp=new Set((viz.packing.unplaced||[]).map(u=>u.instance_id));
  for(const o of viz.instances){
    const col="#"+(PALETTE[o.id%PALETTE.length]).toString(16).padStart(6,"0");
    let box,cls;
    if(o.kept===false){box="dropped";cls="drop";}
    else if(unp.has(o.id)){box="LOOSE (no fit)";cls="nofit";}
    else {box=boxOf[o.id]||"-";cls="fit";}
    const flags=`${o.foldable?'<span class="tag f">fold</span>':''}${o.compressible?'<span class="tag c">compress</span>':''}${crammed.has(o.id)?'<span class="tag x">crammed</span>':''}`;
    h+=`<tr><td><span class="dot" style="background:${col}"></span></td>
      <td>${o.id}</td><td>${o.identity||o.label}</td><td>${o.material||"-"}</td>
      <td>${o.dims_cm.join(" × ")}</td><td>${(o.corrected_dims_cm||o.dims_cm).join(" × ")}</td>
      <td class="${cls}">${box}</td><td>${flags}</td>
      <td class="muted">${o.reasoning||""}</td></tr>`;
  }
  t.innerHTML=h;
  const nb=(viz.packing.boxes||[]).length, nu=(viz.packing.unplaced||[]).length;
  const kept=viz.instances.filter(o=>o.kept!==false).length;
  const tp=viz.truck_plan||{};
  const won=v=>(v||0).toLocaleString("ko-KR")+"원";
  let quote;
  if(tp.oversize) quote=`<span class="nofit">트럭 초과 화물: ${(tp.oversize_items||[]).join(", ")}</span>`;
  else if(tp.truck) quote=
    `<b>${tp.truck.name}</b> x ${tp.count}대 · 적재율 ${Math.round((tp.utilization||0)*100)}%`+
    ` · 총 짐 ${tp.cargo_volume_m3} m³ · 견적 <b class="price">${won(tp.quote_krw)}</b>`;
  else quote=`<span class="muted">트럭 견적 없음</span>`;
  const alts=(tp.options||[]).length
    ? `<div class="alts">대안 톤수: `+
      tp.options.map(o=>`${o.name}×${o.count} <b>${won(o.total_krw)}</b>`).join(" · ")+`</div>`
    : "";
  document.getElementById("pack-summary").innerHTML=
    `<div class="quote">${quote}</div>${alts}`+
    `<div class="meta"><b>${kept}</b> objects → <b>${nb}</b> box(es)`+
    (nu?` · <span class="nofit">${nu} loose (furniture)</span>`:"")+
    ` · VLM: <b>${viz.vlm_source}</b> · seg: ${viz.method}`+
    (viz.frames>1?` · ${viz.frames} frames`:"")+`</div>`;
}

/* ---- toggles ---- */
function applyToggles(){
  const sp=document.getElementById("t-points").checked;
  const sb=document.getElementById("t-obb").checked;
  const sl=document.getElementById("t-labels").checked;
  const sg=document.getElementById("t-grid").checked;
  for(const v of views){
    v.content.children.forEach(c=>{
      if(c.userData.kind==="points") c.visible=sp;
      if(c.userData.kind==="obb"||c.userData.kind==="fill") c.visible=sb;
    });
    v.labels.visible=sl; v.labelRenderer.domElement.style.display=sl?"":"none";
    v.grid.visible=sg;
  }
}
["t-points","t-obb","t-labels","t-grid"].forEach(id=>
  document.getElementById(id).addEventListener("change",applyToggles));

/* ---- run (SSE) ---- */
let CURRENT=null, CURRENT_NAME=null;
function setStatus(txt,frac){
  document.getElementById("status-text").textContent=txt;
  if(frac!=null) document.getElementById("bar-fill").style.width=(frac*100)+"%";
}
async function run(name){
  const catalog=await saveCatalog();
  if(!catalog.length){ setStatus("add at least one box",0); return; }
  setStatus("starting…",0.02);
  const resp=await fetch(`/api/run/${encodeURIComponent(name)}`,{
    method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({catalog})});
  const reader=resp.body.getReader(); const dec=new TextDecoder(); let buf="";
  while(true){
    const {value,done}=await reader.read(); if(done) break;
    buf+=dec.decode(value,{stream:true});
    let idx;
    while((idx=buf.indexOf("\n\n"))>=0){
      const line=buf.slice(0,idx).trim(); buf=buf.slice(idx+2);
      if(!line.startsWith("data:")) continue;
      const evt=JSON.parse(line.slice(5).trim());
      if(evt.type==="progress") setStatus(evt.message,evt.progress);
      else if(evt.type==="done"){ CURRENT=evt.result; CURRENT_NAME=name; setStatus("done",1.0);
        renderLeft(CURRENT); renderRight(CURRENT); renderTable(CURRENT); fillQuoteTotal(); }
      else if(evt.type==="error"){ setStatus("error: "+evt.error,0); console.error(evt.trace); }
    }
  }
}
document.getElementById("btn-run").addEventListener("click",()=>{
  const name=document.getElementById("folder").value; if(name) run(name);
});

/* load a cached result (so the shipped demo renders without re-running SAM2) */
async function loadCached(name){
  try{
    const r=await fetch(`/api/result/${encodeURIComponent(name)}`);
    if(!r.ok) return false;
    CURRENT=await r.json(); CURRENT_NAME=name;
    renderLeft(CURRENT); renderRight(CURRENT); renderTable(CURRENT); fillQuoteTotal();
    setStatus("loaded cached result",1.0); return true;
  }catch(e){ return false; }
}
const folderSel=document.getElementById("folder");
folderSel.addEventListener("change",()=>loadCached(folderSel.value));
if(folderSel.value) loadCached(folderSel.value);

/* ---- config: model paths (auto-filled) + Gemini key (persisted) ---- */
function applyConfig(cfg){
  const back={gemini:"Gemini API",gemma:"local Gemma",heuristic:"heuristic (no model)",
              gemma_e4b:"Gemma 4 E4B",gemma_31b:"Gemma 4 31B"};
  const vs=document.getElementById("vlm-status");
  const active=cfg.active_backend||"heuristic";
  vs.textContent="VLM: "+(back[active]||active);
  vs.className="gemini "+(active==="heuristic"?"off":"on");
  document.getElementById("cfg-backend").value=cfg.vlm_backend||"auto";
  document.getElementById("cfg-sam").textContent=cfg.sam2_ckpt||"— (run setup_models.sh)";
  document.getElementById("cfg-e4b").textContent=cfg.gemma_e4b||"— (run setup_models.sh)";
  document.getElementById("cfg-31b").textContent=cfg.gemma_31b||"— (run setup_models.sh)";
  document.getElementById("cfg-key-state").textContent=
    cfg.gemini_key_set?("saved key "+cfg.gemini_key_hint):"no key saved";
}
async function loadConfig(){
  try{ applyConfig(await (await fetch("/api/config")).json()); }catch(e){}
}
async function saveConfig(updates){
  const r=await fetch("/api/config",{method:"POST",
    headers:{"Content-Type":"application/json"},body:JSON.stringify(updates)});
  if(r.ok) applyConfig(await r.json());
}
document.getElementById("cfg-save").addEventListener("click",()=>{
  const key=document.getElementById("cfg-key").value;
  saveConfig({gemini_api_key:key}).then(()=>{document.getElementById("cfg-key").value="";});
});
document.getElementById("cfg-backend").addEventListener("change",e=>
  saveConfig({vlm_backend:e.target.value}));
loadConfig();

/* ---- editable box catalog (left sidebar, persisted in config.local.json) ---- */
const boxList=document.getElementById("box-list");
let catalogSaveTimer=null;
function scheduleSaveCatalog(){
  clearTimeout(catalogSaveTimer);
  catalogSaveTimer=setTimeout(()=>{ saveCatalog(); },400);
}
async function saveCatalog(){
  const catalog=readBoxes();
  if(!catalog.length) return catalog;
  try{
    const r=await fetch("/api/catalog",{method:"POST",
      headers:{"Content-Type":"application/json"},body:JSON.stringify({catalog})});
    if(r.ok) return await r.json();
  }catch(e){}
  return catalog;
}
function addBoxRow(name="",dims=["","",""]){
  const row=document.createElement("div"); row.className="box-row";
  row.innerHTML=
    `<input class="bn" placeholder="name" value="${name}">`+
    `<input class="bd" type="number" min="1" placeholder="W" value="${dims[0]}">`+
    `<span class="x">×</span><input class="bd" type="number" min="1" placeholder="H" value="${dims[1]}">`+
    `<span class="x">×</span><input class="bd" type="number" min="1" placeholder="D" value="${dims[2]}">`+
    `<button class="del" title="delete">x</button>`;
  row.querySelector(".del").addEventListener("click",()=>{
    row.remove(); scheduleSaveCatalog();
  });
  row.querySelectorAll("input").forEach(i=>i.addEventListener("input",scheduleSaveCatalog));
  boxList.appendChild(row);
}
function readBoxes(){
  const out=[];
  boxList.querySelectorAll(".box-row").forEach(r=>{
    const name=r.querySelector(".bn").value.trim();
    const d=[...r.querySelectorAll(".bd")].map(i=>parseFloat(i.value));
    if(d.every(v=>v>0)) out.push({name,dims_cm:d});
  });
  return out;
}
document.getElementById("btn-add-box").addEventListener("click",()=>{ addBoxRow(); scheduleSaveCatalog(); });
(async function loadCatalog(){
  try{
    const r=await fetch("/api/catalog"); const cat=await r.json();
    if(Array.isArray(cat)&&cat.length) cat.forEach(b=>addBoxRow(b.name,b.dims_cm));
    else addBoxRow();
  }catch(e){ addBoxRow(); }
})();

/* ---- multi-frame upload rows (a house = several RGB-D shots) ---- */
const frameRows=document.getElementById("frame-rows");
function addFrameRow(){
  const n=frameRows.children.length;
  const row=document.createElement("div"); row.className="frow";
  row.innerHTML=
    `<span class="flabel">frame ${n+1}</span>`+
    `<input type="file" class="fr-rgb" accept="image/png" title="rgb.png">`+
    `<input type="file" class="fr-dep" accept=".npy" title="depth_aligned.npy">`+
    `<input type="file" class="fr-rep" accept=".json" title="report.json (optional)">`+
    `<button class="del" title="remove">x</button>`;
  row.querySelector(".del").addEventListener("click",()=>{ row.remove(); relabelFrames(); });
  frameRows.appendChild(row);
}
function relabelFrames(){
  [...frameRows.children].forEach((r,i)=>r.querySelector(".flabel").textContent="frame "+(i+1));
}
document.getElementById("btn-add-frame").addEventListener("click",addFrameRow);
addFrameRow();   // start with one row

document.getElementById("btn-upload").addEventListener("click",async()=>{
  const fd=new FormData();
  const z=document.getElementById("up-zip").files[0];
  if(z){ fd.append("zip",z); fd.append("name",z.name.replace(/\.zip$/i,"")); }
  else{
    let pairs=0;
    for(const row of frameRows.children){
      const rgb=row.querySelector(".fr-rgb").files[0];
      const dep=row.querySelector(".fr-dep").files[0];
      const rep=row.querySelector(".fr-rep").files[0];
      if(!rgb||!dep) continue;            // skip empty rows
      fd.append("rgb",rgb); fd.append("depth",dep);
      fd.append("report", rep || new Blob([]), rep?rep.name:"");
      pairs++;
    }
    if(!pairs){ setStatus("each frame needs rgb + depth",0); return; }
    fd.append("name","upload_"+(frameRows.children.length>1?"scene_":"")+Date.now());
  }
  setStatus("uploading…",0.05);
  const r=await fetch("/api/upload",{method:"POST",body:fd});
  const j=await r.json();
  if(!r.ok){ setStatus("error: "+(j.error||"upload failed"),0); return; }
  const sel=document.getElementById("folder");
  const lbl=j.name+" · upload"+(j.frames>1?` · ${j.frames} frames`:"");
  const opt=document.createElement("option"); opt.value=j.name; opt.textContent=lbl;
  sel.appendChild(opt); sel.value=j.name;
  setStatus(`uploaded: ${j.name} (${j.frames||1} frame${j.frames>1?"s":""})`,0.1);
});

/* ---- 견적서 생성 (fill the .docx quote form) ---- */
const Q_FIELDS=["manager","customer_name","customer_phone","origin_address",
  "origin_date","origin_time","dest_address","dest_date","dest_time",
  "work_in_type","work_out_type","crew_in","crew_out","cost_in","cost_out",
  "deposit","total","storage_period","storage_total","storage_daily"];
function fillQuoteTotal(){
  // pre-fill the quote form from the VLM estimate (only empty inputs, so edits persist)
  const est=(CURRENT&&CURRENT.quote_estimate)||{};
  const set=(id,v)=>{ const el=document.getElementById(id);
    if(el&&v!=null&&v!==""&&!el.value) el.value=v; };
  set("q-work_in_type",est.work_in_type); set("q-work_out_type",est.work_out_type);
  set("q-crew_in",est.crew_in); set("q-crew_out",est.crew_out);
  set("q-cost_in",est.cost_in); set("q-cost_out",est.cost_out);
  set("q-deposit",est.deposit); set("q-total",est.total);
  if(Array.isArray(est.requests)&&est.requests.length){
    const t=document.getElementById("q-requests"); if(t&&!t.value.trim()) t.value=est.requests.join("\n");
  }
  // also surface the pipeline's truck fare as a 합계 placeholder
  const tp=CURRENT&&CURRENT.truck_plan; const tot=document.getElementById("q-total");
  if(tp&&tp.quote_krw&&tot&&!tot.value) tot.placeholder=Math.round(tp.quote_krw/10000)+" (자동)";
}
function readQuoteForm(){
  const d={};
  for(const f of Q_FIELDS){
    const v=document.getElementById("q-"+f).value.trim();
    if(v) d[f]=v;
  }
  const lines=id=>document.getElementById(id).value.split("\n").map(s=>s.trim()).filter(Boolean);
  const req=lines("q-requests"); if(req.length) d.requests=req;
  const dis=lines("q-disposal"); if(dis.length) d.disposal=dis;
  return d;
}
document.getElementById("btn-quote").addEventListener("click",async()=>{
  const st=document.getElementById("q-state");
  if(!CURRENT_NAME){ st.textContent="먼저 파이프라인을 실행하세요"; return; }
  st.textContent="생성 중…";
  try{
    const r=await fetch(`/api/quote/${encodeURIComponent(CURRENT_NAME)}`,{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(readQuoteForm())});
    if(!r.ok){ const j=await r.json().catch(()=>({})); st.textContent="오류: "+(j.error||r.status); return; }
    const blob=await r.blob();
    const fname=`견적서_${CURRENT_NAME}.docx`;
    const url=URL.createObjectURL(blob);
    const a=document.createElement("a"); a.href=url; a.download=fname; a.click();
    URL.revokeObjectURL(url);
    st.textContent="다운로드됨: "+fname;
  }catch(e){ st.textContent="오류: "+e.message; }
});
