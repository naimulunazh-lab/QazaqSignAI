const API='/api';
let user=null;
const token=()=>localStorage.getItem('qazaqsign_token');
async function request(path,options={}){const response=await fetch(`${API}${path}`,{...options,headers:{'Content-Type':'application/json',...(token()?{Authorization:`Bearer ${token()}`}:{})}});const data=await response.json().catch(()=>({}));if(!response.ok)throw new Error(data.error||'Серверге қосылу мүмкін болмады.');return data;}
export const account={
 async restore(){if(!token())return null;try{user=(await request('/auth/me')).user;return user}catch{localStorage.removeItem('qazaqsign_token');return null;}},
 async signIn(email,password,register,name){const route=register?'/auth/register':'/auth/login';const data=await request(route,{method:'POST',body:JSON.stringify({email,password,name})});localStorage.setItem('qazaqsign_token',data.token);user=data.user;return user;},
 current(){return user;},
 async saveProgress(attempt){if(!user)return;await request(`/progress/${encodeURIComponent(attempt.lessonId)}`,{method:'PUT',body:JSON.stringify(attempt)});},
 async getProgress(){if(!user)return [];return (await request('/progress')).items;},
 signOut(){localStorage.removeItem('qazaqsign_token');user=null;}
};
export function mountAccount(onChange){
 const button=document.createElement('button');button.className='account-button';button.textContent='Кіру';document.querySelector('.streak').before(button);
 document.body.insertAdjacentHTML('beforeend',`<dialog id="authDialog" class="auth-dialog"><div class="auth-card"><button class="modal-close" type="button">×</button><p class="eyebrow">QAZAQSIGN AI</p><h2 id="authTitle">Оқуыңды сақта</h2><p>Кез келген құрылғыдан прогреске орал.</p><div class="auth-switch"><button type="button" class="active" data-mode="login">Кіру</button><button type="button" data-mode="register">Тіркелу</button></div><form id="authForm"><label id="nameField" hidden>Аты-жөнің<input name="name" maxlength="80"></label><label>Email<input required name="email" type="email" autocomplete="email"></label><label>Құпиясөз<input required name="password" type="password" minlength="8" autocomplete="current-password"></label><p id="authError" class="auth-error"></p><button class="button primary" id="authSubmit">Кіру</button></form></div></dialog>`);
 const dialog=$('#authDialog'),form=$('#authForm');let register=false;const refresh=()=>{button.textContent=user?user.name:'Кіру';onChange?.(user);};
 button.onclick=()=>{if(user){account.signOut();refresh();return;}dialog.showModal();};dialog.querySelector('.modal-close').onclick=()=>dialog.close();
 dialog.querySelectorAll('[data-mode]').forEach(tab=>tab.onclick=()=>{register=tab.dataset.mode==='register';dialog.querySelectorAll('[data-mode]').forEach(item=>item.classList.toggle('active',item===tab));$('#nameField').hidden=!register;$('#authTitle').textContent=register?'Жаңа жолды баста':'Оқуыңды сақта';$('#authSubmit').textContent=register?'Тіркелу':'Кіру';});
 form.onsubmit=async event=>{event.preventDefault();const values=new FormData(form);$('#authError').textContent='';try{await account.signIn(values.get('email'),values.get('password'),register,values.get('name'));dialog.close();refresh();}catch(error){$('#authError').textContent=error.message;}};account.restore().then(refresh);
}
const $=selector=>document.querySelector(selector);
