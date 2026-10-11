(() => {
"use strict";
// The Worker origin is public; secrets remain exclusively in Cloudflare.
const base=String(window.BBB_AUTH_API_BASE||"https://bbb-license-gateway.zliu53468.workers.dev").replace(/\/+$/,"");
const byId=id=>document.getElementById(id),note=byId("note"),tbody=byId("records");
function say(text){note.textContent=text;}
async function call(path,method="GET",body){
  if(!base.startsWith("https://"))throw new Error("請先在 auth-config.js 填入 HTTPS Worker 網址。");
  const token=byId("adminToken").value.trim();
  if(token.length<24)throw new Error("管理者金鑰不得少於 24 碼。");
  const r=await fetch(base+path,{method,cache:"no-store",headers:{"Authorization":"Bearer "+token,...(body?{"Content-Type":"application/json"}:{})},body:body?JSON.stringify(body):undefined});
  const result=await r.json().catch(()=>({error:"invalid_server_response"}));
  if(!r.ok)throw new Error(result.error||"server_error_"+r.status);
  return result;
}
function td(text){const cell=document.createElement("td");cell.textContent=String(text??"—");return cell;}
function date(ms){return ms?new Date(ms).toLocaleString("zh-TW"):"—";}
async function load(){
  say("讀取中…");
  const data=await call("/api/admin/licenses");
  tbody.textContent="";
  for(const row of data.licenses){
    const tr=document.createElement("tr");
    const status=row.revoked_at?"已停用":row.expires_at&&row.expires_at<=Date.now()?"已到期":row.bound_at?"已綁定":"尚未綁定";
    for(const value of [row.label||"—","••••"+row.code_suffix,status,date(row.bound_at),date(row.expires_at)])tr.appendChild(td(value));
    const actions=document.createElement("td");
    for(const [action,label,klass] of [["reset","重置裝置","secondary"],["revoke","停用","danger"]]){
      const b=document.createElement("button");b.type="button";b.textContent=label;b.className=klass;b.disabled=!!row.revoked_at;
      b.addEventListener("click",async()=>{
        if(!confirm(action==="reset"?"要解除這組碼的現有裝置綁定嗎？原裝置重新開通前，其他持碼者可搶先綁定。":"確定停用此開通碼？這個動作不能直接復原。"))return;
        try{b.disabled=true;await call("/api/admin/licenses/"+encodeURIComponent(row.id)+"/"+action,"POST",{});await load();}
        catch(e){say("操作失敗："+e.message);b.disabled=false;}
      });actions.appendChild(b);
    }
    tr.appendChild(actions);tbody.appendChild(tr);
  }
  if(!data.licenses.length){const tr=document.createElement("tr"),cell=td("尚無開通碼");cell.colSpan=6;tr.appendChild(cell);tbody.appendChild(tr);}
  say("目前共有 "+data.licenses.length+" 組開通碼（最多顯示最近 250 組）。");
}
byId("refresh").addEventListener("click",()=>load().catch(e=>say("讀取失敗："+e.message)));
byId("create").addEventListener("click",async()=>{
  const count=Number(byId("count").value),label=byId("label").value.trim(),dateValue=byId("expiresAt").value;
  if(!Number.isInteger(count)||count<1||count>30){say("每次限 1～30 組。");return;}
  let expires_at=null;
  if(dateValue){expires_at=new Date(dateValue+"T23:59:59+08:00").getTime();if(!Number.isFinite(expires_at)||expires_at<=Date.now()){say("到期日必須在未來。");return;}}
  const button=byId("create");button.disabled=true;
  try{
    const data=await call("/api/admin/licenses","POST",{count,label,expires_at});
    byId("created").textContent=data.created.map((r,i)=>(i+1)+". "+r.code).join("\n")+"\n\n請立即複製並保存，稍後無法重新查詢完整開通碼。";
    await load();say("已建立 "+data.created.length+" 組開通碼，請保存。");
  }catch(e){say("建立失敗："+e.message);}
  finally{button.disabled=false;}
});
})();
