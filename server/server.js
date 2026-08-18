require('dotenv').config();
const path=require('path');
const express=require('express'); const cors=require('cors'); const helmet=require('helmet');
const rateLimit=require('express-rate-limit'); const bcrypt=require('bcryptjs'); const jwt=require('jsonwebtoken'); const {Pool}=require('pg');
const app=express(), port=Number(process.env.PORT||3000);
if(!process.env.DATABASE_URL||!process.env.JWT_SECRET) throw new Error('DATABASE_URL and JWT_SECRET must be set.');
const pool=new Pool({connectionString:process.env.DATABASE_URL,ssl:process.env.DATABASE_SSL==='true'?{rejectUnauthorized:false}:false});
const origins=(process.env.ALLOWED_ORIGINS||'').split(',').filter(Boolean);
app.use(helmet({
  crossOriginResourcePolicy:{policy:'cross-origin'},
  contentSecurityPolicy:{directives:{
    "script-src":["'self'","'wasm-unsafe-eval'"],
    "style-src":["'self'","https://fonts.googleapis.com"],
    "font-src":["'self'","https://fonts.gstatic.com"],
    "frame-src":["'self'","https://www.youtube-nocookie.com","https://www.youtube.com"]
  }}
}));
app.use(cors({origin(origin,cb){if(!origin||origins.includes(origin))return cb(null,true);cb(new Error('Origin is not allowed.'));}}));
app.use(express.json({limit:'32kb'}));
app.use('/api/auth',rateLimit({windowMs:15*60*1000,limit:20,standardHeaders:true,legacyHeaders:false}));
app.use('/vendor/mediapipe/holistic',express.static(path.join(__dirname,'node_modules','@mediapipe','holistic')));
app.use('/vendor/mediapipe/camera_utils',express.static(path.join(__dirname,'node_modules','@mediapipe','camera_utils')));
app.use('/vendor/mediapipe/drawing_utils',express.static(path.join(__dirname,'node_modules','@mediapipe','drawing_utils')));
app.use('/vendor/tfjs',express.static(path.join(__dirname,'node_modules','@tensorflow','tfjs','dist')));
const email=value=>String(value||'').trim().toLowerCase();
const publicUser=row=>({id:row.id,name:row.name,email:row.email});
const issue=user=>jwt.sign({sub:user.id},process.env.JWT_SECRET,{expiresIn:'7d',issuer:'qazaqsign-ai'});
async function auth(req,res,next){try{const value=req.headers.authorization||'';const token=value.startsWith('Bearer ')?value.slice(7):null;if(!token)return res.status(401).json({error:'Тіркелгіге кіріңіз.'});const claim=jwt.verify(token,process.env.JWT_SECRET,{issuer:'qazaqsign-ai'});const result=await pool.query('SELECT id,name,email FROM users WHERE id=$1',[claim.sub]);if(!result.rowCount)return res.status(401).json({error:'Тіркелгі табылмады.'});req.user=result.rows[0];next();}catch{return res.status(401).json({error:'Сессия мерзімі аяқталды.'});}}
app.post('/api/auth/register',async(req,res,next)=>{try{const name=String(req.body.name||'').trim(),mail=email(req.body.email),password=String(req.body.password||'');if(name.length<2||name.length>80||!/^\S+@\S+\.\S+$/.test(mail)||password.length<8)return res.status(400).json({error:'Аты, email және кемінде 8 таңбалы құпиясөзді енгізіңіз.'});const hash=await bcrypt.hash(password,12);const result=await pool.query('INSERT INTO users(name,email,password_hash) VALUES($1,$2,$3) RETURNING id,name,email',[name,mail,hash]);const user=result.rows[0];res.status(201).json({user:publicUser(user),token:issue(user)});}catch(error){if(error.code==='23505')return res.status(409).json({error:'Бұл email тіркелген.'});next(error);}});
app.post('/api/auth/login',async(req,res,next)=>{try{const result=await pool.query('SELECT * FROM users WHERE email=$1',[email(req.body.email)]);const user=result.rows[0];if(!user||!(await bcrypt.compare(String(req.body.password||''),user.password_hash)))return res.status(401).json({error:'Email немесе құпиясөз қате.'});res.json({user:publicUser(user),token:issue(user)});}catch(error){next(error);}});
app.get('/api/auth/me',auth,(req,res)=>res.json({user:publicUser(req.user)}));
app.get('/api/progress',auth,async(req,res,next)=>{try{const result=await pool.query('SELECT lesson_id AS "lessonId",level_id AS "levelId",accuracy,completed,updated_at AS "updatedAt" FROM lesson_progress WHERE user_id=$1 ORDER BY updated_at DESC',[req.user.id]);res.json({items:result.rows});}catch(error){next(error);}});
app.put('/api/progress/:lessonId',auth,async(req,res,next)=>{try{const lessonId=String(req.params.lessonId||'').slice(0,80),levelId=String(req.body.levelId||'').slice(0,80),accuracy=Math.round(Number(req.body.accuracy));if(!lessonId||!levelId||!Number.isFinite(accuracy)||accuracy<0||accuracy>100)return res.status(400).json({error:'Прогресс деректері жарамсыз.'});const completed=req.body.completed===true;const result=await pool.query(`INSERT INTO lesson_progress(user_id,lesson_id,level_id,accuracy,completed) VALUES($1,$2,$3,$4,$5) ON CONFLICT(user_id,lesson_id) DO UPDATE SET accuracy=GREATEST(lesson_progress.accuracy,EXCLUDED.accuracy),completed=lesson_progress.completed OR EXCLUDED.completed,updated_at=NOW() RETURNING lesson_id AS "lessonId",accuracy,completed`,[req.user.id,lessonId,levelId,accuracy,completed]);res.json({item:result.rows[0]});}catch(error){next(error);}});
app.get('/api/health',async(req,res)=>{try{await pool.query('SELECT 1');res.json({status:'ok'});}catch{res.status(503).json({status:'database unavailable'});}});
app.use(express.static(path.join(__dirname,'..'),{index:'index.html'}));
app.use((error,req,res,next)=>{console.error(error);res.status(500).json({error:'Ішкі сервер қатесі.'});});
app.listen(port,()=>console.log(`QazaqSign AI is listening on ${port}`));
