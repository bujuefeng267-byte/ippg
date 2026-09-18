'use strict';
// Exercise the shipped HTML's actual script, not a second implementation.
// Deterministic browser mocks model rVFC, canvas snapshot-at-toBlob invocation,
// permission promises, and independent HTTP completion. No camera is opened.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const flush = async () => { for (let i = 0; i < 30; i++) await Promise.resolve(); };
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); return {promise, resolve, reject}; };

function browser(options = {}) {
  let now = 0, nextCallback = 0, nextSession = 0, activeSession = options.activeSession || null;
  let activeFrames = 0, peakFrames = 0, canvasSnapshot = null;
  const callbacks = new Map(), cancelledCallbacks = [], calls = [], frames = [], blobs = [], tracks = [], listeners = {};
  const draw = {scale(){}, clearRect(){}, fillText(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){},
    drawImage(video) { canvasSnapshot = {frame:video.frame, media:video.currentTime}; },
    getImageData() { return {data:{...canvasSnapshot, kind:'rgba'}}; }};
  const elements = Object.fromEntries(['start','stop','camera','transport','preview','capture','wave','hr','message','error','saved','fps','cost','lag','bpm','quality','window'].map(id => [id, {
    id, value:id === 'transport' ? 'jpeg' : '', disabled:false, textContent:'', innerHTML:'', width:960, height:540,
    getContext:() => draw, getBoundingClientRect:() => ({width:400,height:190}), appendChild(){},
  }]));
  Object.assign(elements.preview, {
    videoWidth:960, videoHeight:540, currentTime:0, frame:null, srcObject:null,
    play:async () => { if(options.playError)throw options.playError; },
    requestVideoFrameCallback(fn){const id = ++nextCallback; callbacks.set(id, fn); return id;},
    cancelVideoFrameCallback(id){if(callbacks.has(id))cancelledCallbacks.push(callbacks.get(id));callbacks.delete(id);},
  });
  elements.capture.toBlob = (callback, type, quality) => {
    const snapshot = {...canvasSnapshot, kind:'jpeg', type, quality};
    const pending = {snapshot, complete:() => callback(snapshot)};
    blobs.push(pending);
    if(!options.manualBlob)pending.complete();
  };
  const makeStream = () => {
    const track = {stopped:false, stop(){this.stopped = true;}, getSettings:() => ({deviceId:'camera1'})};
    tracks.push(track); return {getTracks:() => [track], getVideoTracks:() => [track]};
  };
  const response = (data, ok=true) => ({ok,statusText:ok?'OK':'Conflict',json:async()=>data});
  async function fetch(url, init={}) {
    const call = {url, init, at:now};calls.push(call);
    if(url === '/api/status')return response({active:Boolean(activeSession),session:activeSession,latest:null});
    if(url === '/api/start') {
      if(options.startHook) { const result = await options.startHook(call, ++nextSession); if(result.ok!==false)activeSession=result.session;return response(result,result.ok!==false); }
      activeSession = 'session'+(++nextSession);return response({session:activeSession});
    }
    if(url === '/api/stop') {if(activeSession === init.headers['X-Session'])activeSession=null;return response({stopped:true});}
    if(url === '/api/frame') {
      activeFrames++;peakFrames=Math.max(peakFrames,activeFrames);
      const pending=deferred();const frame={...call,pending,complete:(ok=true)=>pending.resolve(response(ok?{ok:true}:{error:'old session'},ok))};
      frames.push(frame);try{return await pending.promise;}finally{activeFrames--;}
    }
    throw Error('Unexpected URL '+url);
  }
  const context = vm.createContext({
    document:{getElementById:id=>elements[id],createElement:()=>({})},
    navigator:{mediaDevices:{
      getUserMedia:async()=>options.getUserMedia?options.getUserMedia(makeStream):makeStream(),
      enumerateDevices:async()=>{if(options.enumerateError)throw Error('enumeration unavailable');return [{kind:'videoinput',deviceId:'camera1',label:'Camera'}];},
    }},
    window:{addEventListener:(name,fn)=>{listeners[name]=fn;}},
    performance:{now:()=>now},devicePixelRatio:1,fetch,setInterval:()=>1,
    requestAnimationFrame:fn=>{const id=++nextCallback;callbacks.set(id,fn);return id;},cancelAnimationFrame:id=>callbacks.delete(id),
  });
  vm.runInContext(script, context, {filename:'index.reviewed.html'});
  return {
    elements,calls,frames,blobs,tracks,callbacks,cancelledCallbacks,listeners,context,
    get peakFrames(){return peakFrames;}, get activeFrames(){return activeFrames;},
    state:expr=>vm.runInContext(expr,context),
    time(value){now=value;},
    async start(){await elements.start.onclick();await flush();},
    async stop(){await elements.stop.onclick();await flush();},
    async capture(media, frame, at){
      now=at;elements.preview.currentTime=media;elements.preview.frame=frame;
      const entry=callbacks.entries().next().value;assert.ok(entry,'one capture callback must remain scheduled');
      callbacks.delete(entry[0]);entry[1](at,{mediaTime:media});await flush();
    },
    async complete(index, at, ok=true){now=at;frames[index].complete(ok);await flush();},
  };
}

const cases=[];
function test(name,fn){cases.push({name,fn});}
test('JPEG 98 is the default option and encoding quality', async()=>{
  assert.match(html, /id="transport"[^>]*><option value="jpeg">/);
  const b=browser();await flush();await b.start();await b.capture(10,'first',0);
  assert.equal(b.frames[0].init.headers['Content-Type'],'image/jpeg');assert.equal(b.frames[0].init.body.quality,.98);
});
test('permission failure never stops an observed session',async()=>{
  const b=browser({activeSession:'other',getUserMedia:()=>{const e=Error('denied');e.name='NotAllowedError';throw e;}});
  await flush();await b.start();assert.equal(b.calls.filter(x=>x.url==='/api/stop').length,0);assert.equal(b.state('ownsSession'),false);
});
test('start failure cleans tracks without stopping another session',async()=>{
  const b=browser({activeSession:'other',startHook:()=>({ok:false,error:'another session is active'})});
  await flush();await b.start();assert.equal(b.calls.filter(x=>x.url==='/api/stop').length,0);assert.equal(b.tracks[0].stopped,true);
});
test('late camera permission after stop is released without starting the server',async()=>{
  const permission=deferred();let make;
  const b=browser({getUserMedia:f=>{make=f;return permission.promise;}});await flush();
  const start=b.start();await flush();await b.stop();permission.resolve(make());await start;
  assert.equal(b.tracks[0].stopped,true);assert.equal(b.calls.filter(x=>x.url==='/api/start').length,0);
});
test('late start response stops only its exact orphaned ID',async()=>{
  const first=deferred();const b=browser({startHook:(_,n)=>n===1?first.promise:{session:'new'}});await flush();
  const oldStart=b.start();await flush();await b.stop();await b.start();first.resolve({session:'orphan'});await oldStart;
  assert.equal(b.state('session'),'new');assert.equal(b.state('running'),true);
  assert.deepEqual(b.calls.filter(x=>x.url==='/api/stop').map(x=>x.init.headers['X-Session']),['orphan']);
});
test('independent capture/pump sends the 33 ms frame at 35 ms, not 66 ms',async()=>{
  const b=browser();await flush();await b.start();await b.capture(10,'zero',0);await b.capture(10.033,'thirty-three',33);
  assert.equal(b.frames.length,1);await b.complete(0,35);
  assert.equal(b.frames.length,2);assert.equal(b.frames[1].at,35);assert.equal(b.frames[1].init.body.frame,'thirty-three');assert.equal(b.peakFrames,1);
});
test('backpressure replaces the pending canvas with only the latest frame',async()=>{
  const b=browser();await flush();await b.start();await b.capture(20,'zero',0);
  await b.capture(20.033,'one',33);await b.capture(20.066,'two',66);await b.capture(20.099,'three',99);await b.complete(0,110);
  assert.deepEqual(b.frames.map(f=>f.init.body.frame),['zero','three']);assert.equal(b.peakFrames,1);
  assert.ok(Math.abs(Number(b.frames[1].init.headers['X-Capture-Time'])-.099)<1e-12);
});
test('asynchronous JPEG uses the timestamp of its own frozen canvas snapshot',async()=>{
  const b=browser({manualBlob:true});await flush();await b.start();await b.capture(12,'older',0);await b.capture(12.033,'newer',33);
  b.time(35);b.blobs[0].complete();await flush();assert.equal(b.frames[0].init.body.frame,'older');assert.equal(b.frames[0].init.headers['X-Capture-Time'],'0');
  await b.complete(0,70);assert.equal(b.blobs[1].snapshot.frame,'newer');b.blobs[1].complete();await flush();
  assert.equal(b.frames[1].init.body.frame,'newer');assert.ok(Math.abs(Number(b.frames[1].init.headers['X-Capture-Time'])-.033)<1e-12);
});
test('RGBA snapshot preserves frame/time and dimension headers',async()=>{
  const b=browser();b.elements.transport.value='rgba';await flush();await b.start();await b.capture(2,'rgba-first',0);
  await b.capture(2.05,'rgba-second',50);await b.complete(0,60);
  assert.equal(b.frames[0].init.body.frame,'rgba-first');assert.equal(b.frames[1].init.body.frame,'rgba-second');
  assert.equal(b.frames[1].init.headers['X-Frame-Width'],'960');assert.equal(b.frames[1].init.headers['X-Frame-Height'],'540');
  assert.ok(Math.abs(Number(b.frames[1].init.headers['X-Capture-Time'])-.05)<1e-12);
});
test('old JPEG encoding completion cannot upload to a restarted session',async()=>{
  const b=browser({manualBlob:true});await flush();await b.start();await b.capture(5,'old',0);await b.stop();await b.start();await b.capture(9,'new',100);
  assert.equal(b.blobs.length,1,'old encoder retains the sole operation slot');b.blobs[0].complete();await flush();
  assert.equal(b.frames.length,0,'old encoding must never be uploaded');assert.equal(b.blobs.length,2,'slot releases immediately to latest current frame');
  b.blobs[1].complete();await flush();assert.equal(b.frames[0].init.headers['X-Session'],'session2');assert.equal(b.frames[0].init.body.frame,'new');
});
test('stop/restart never overlaps old/new HTTP or changes destination IDs',async()=>{
  const b=browser();await flush();await b.start();await b.capture(5,'old',0);const cancelled=[...b.callbacks.values()][0];
  await b.stop();await b.start();await b.capture(9,'new',100);
  assert.equal(b.frames.length,1);assert.equal(b.activeFrames,1);cancelled(110,{mediaTime:5.033});await flush();
  assert.equal(b.state('latestFrame.session'),'session2');await b.complete(0,120,false);
  assert.equal(b.frames.length,2);assert.equal(b.peakFrames,1);
  assert.deepEqual(b.frames.map(f=>f.init.headers['X-Session']),['session1','session2']);assert.deepEqual(b.frames.map(f=>f.init.body.frame),['old','new']);
  assert.equal(b.state('running'),true,'old rejected response must not stop the new camera');
});
test('optional device enumeration failure leaves the camera running',async()=>{
  const b=browser({enumerateError:true});await flush();await b.start();assert.equal(b.state('running'),true);assert.equal(b.tracks[0].stopped,false);
  assert.equal(b.calls.filter(x=>x.url==='/api/stop').length,0);
});
test('monitor unload cannot stop another tab; owner unload stops its own ID',async()=>{
  const monitor=browser({activeSession:'other'});await flush();monitor.listeners.beforeunload();await flush();
  assert.equal(monitor.calls.filter(x=>x.url==='/api/stop').length,0);
  const owner=browser();await flush();await owner.start();owner.listeners.beforeunload();await flush();
  assert.deepEqual(owner.calls.filter(x=>x.url==='/api/stop').map(x=>x.init.headers['X-Session']),['session1']);assert.equal(owner.tracks[0].stopped,true);
});

(async()=>{
  const results=[];
  for(const {name,fn} of cases){await fn();results.push({name,passed:true});console.log('PASS '+name);}
  const report={script_source:'index.html actual script',tests_passed:results.length,cases:results,
    simulated_schedule:{capture_ms:[0,33],http_complete_ms:35,next_upload_ms:35,maximum_parallel_frame_requests:1},
    scope:'Deterministic Node VM browser API mocks; not a physical camera or real-browser performance measurement.'};
  fs.mkdirSync(path.join(__dirname,'diagnostics'),{recursive:true});
  fs.writeFileSync(path.join(__dirname,'diagnostics','browser_lifecycle_test_results.json'),JSON.stringify(report,null,2)+'\n');
  console.log(`${results.length} browser lifecycle tests passed.`);
})().catch(error=>{console.error(error);process.exitCode=1;});
