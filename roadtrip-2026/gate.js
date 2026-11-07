(()=>{
'use strict';
const copy={zh:{title:'输入 PIN',intro:'和朋友一起上路。输入邀请人给你的 PIN，查看完整行程。',label:'8 位数字 PIN',hint:'PIN 请向邀请人获取。',unlock:'解锁行程 →',invalid:'请输入完整的 8 位数字 PIN。',loading:'正在解锁…',network:'页面内容暂时无法加载，请检查网络后重试。',crypto:'请使用支持安全加密的新版浏览器打开 HTTPS 链接。',wrong:'PIN 不正确，请确认后重试。',foot:'行程内容已加密，不保存你的 PIN；刷新后需重新解锁。',home:'返回个人主页',language:'Switch to English'},en:{title:'Enter PIN',intro:'A road trip to share with friends. Enter the PIN from your host to explore the itinerary.',label:'8-digit PIN',hint:'Ask your host for the PIN.',unlock:'Unlock itinerary →',invalid:'Please enter the full 8-digit PIN.',loading:'Unlocking…',network:'Unable to load the itinerary. Check your connection and try again.',crypto:'Open this HTTPS link in a modern browser that supports secure encryption.',wrong:'Incorrect PIN. Please check and try again.',foot:'The itinerary is encrypted. Your PIN is not saved. Refreshing locks the page again.',home:'Back to homepage',language:'切换为中文'}};
const form=document.getElementById('unlock'),field=document.getElementById('pin'),button=document.getElementById('button'),message=document.getElementById('error'),language=document.getElementById('language');
let locale=document.documentElement.lang==='en'?'en':'zh',errorKey='';
const root=new URL(locale==='en'?'../':'./',location.href);
function renderGate(){const t=copy[locale];document.documentElement.lang=locale==='en'?'en':'zh-Hans';document.title='Road Trip 2026 · '+t.title;document.querySelectorAll('[data-copy]').forEach(el=>el.textContent=t[el.dataset.copy]);button.textContent=button.disabled?t.loading:t.unlock;message.textContent=errorKey?t[errorKey]:'';language.textContent=locale==='en'?'中文':'English';language.setAttribute('aria-label',t.language);}
function showError(key){errorKey=key;renderGate();}
language.addEventListener('click',()=>{locale=locale==='en'?'zh':'en';const url=new URL(locale==='en'?'en/':'./',root);url.search=location.search;url.hash=location.hash;history.replaceState(null,'',url);renderGate();});
const bytes=value=>Uint8Array.from(atob(value),c=>c.charCodeAt(0));
function openPages(pages){
 const renderPage=()=>{const english=location.pathname.endsWith('/en/');const doc=pages[english?'en':'zh'];document.open();document.write(doc);document.close();window.addEventListener('popstate',onPop);};
 window.roadtripChangeLanguage=url=>{if(url.origin!==root.origin||![root.pathname,root.pathname+'en/'].includes(url.pathname))return;history.pushState(null,'',url);renderPage();};
 const onPop=()=>{if([root.pathname,root.pathname+'en/'].includes(location.pathname))renderPage();};
 renderPage();
}
form.addEventListener('submit',async event=>{
 event.preventDefault();let pin=field.value.trim();
 if(!/^\d{8}$/.test(pin)){showError('invalid');return;}
 if(!crypto?.subtle){showError('crypto');return;}
 button.disabled=true;showError('');let payload;
 try{const response=await fetch(new URL('content.enc.json',root),{cache:'no-store'});if(!response.ok)throw Error();payload=await response.json();if(payload.version!==1||payload.iterations!==600000)throw Error();}
 catch{pin='';button.disabled=false;showError('network');return;}
 let plain;
 try{const material=await crypto.subtle.importKey('raw',new TextEncoder().encode(pin),'PBKDF2',false,['deriveKey']);pin='';const key=await crypto.subtle.deriveKey({name:'PBKDF2',salt:bytes(payload.salt),iterations:payload.iterations,hash:'SHA-256'},material,{name:'AES-GCM',length:256},false,['decrypt']);plain=await crypto.subtle.decrypt({name:'AES-GCM',iv:bytes(payload.iv)},key,bytes(payload.data));}
 catch{pin='';field.value='';button.disabled=false;showError('wrong');field.focus();return;}
 try{const pages=JSON.parse(new TextDecoder().decode(plain));if(typeof pages.zh!=='string'||typeof pages.en!=='string')throw Error();field.value='';openPages(pages);}
 catch{button.disabled=false;showError('network');}
});
renderGate();
})();
