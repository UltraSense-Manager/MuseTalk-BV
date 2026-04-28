(function () {
  'use strict';

  const PCM_SAMPLE_RATE = 48000;
  const PCM_CHANNELS = 1;
  const PCM_SAMPLE_WIDTH = 2; // 16-bit
  const MAX_DURATION_SEC = 20;
  const TRAIN_CHUNK_MS = 10000;
  // Clone API returns raw PCM at this rate (OpenVoice default)
  const CLONE_OUTPUT_SAMPLE_RATE = 22000;

  function getApiBase() {
    return (document.getElementById('apiBase').value || '').replace(/\/$/, '');
  }

  function getJwt() {
    return (document.getElementById('jwt').value || '').trim();
  }

  function setStateDisplay(text) {
    document.getElementById('stateDisplay').textContent = text;
  }

  function setTrainStatus(msg, isError) {
    const el = document.getElementById('trainStatus');
    el.textContent = msg;
    el.className = 'status' + (isError ? ' error' : '');
  }

  function setCloneStatus(msg, isError) {
    const el = document.getElementById('cloneStatus');
    el.textContent = msg;
    el.className = 'status' + (isError ? ' error' : '');
  }

  function formatStopwatch(ms) {
    const totalSec = ms / 1000;
    const min = Math.floor(totalSec / 60);
    const sec = Math.floor(totalSec % 60);
    const tenth = Math.floor((totalSec % 1) * 10);
    return min + ':' + String(sec).padStart(2, '0') + '.' + tenth;
  }

  let stopwatchStart = null;
  let stopwatchTick = null;

  function startStopwatch() {
    if (stopwatchTick) return;
    stopwatchStart = Date.now();
    const el = document.getElementById('stopwatch');
    el.classList.add('running');
    function tick() {
      if (!stopwatchStart) return;
      el.textContent = formatStopwatch(Date.now() - stopwatchStart);
      stopwatchTick = requestAnimationFrame(tick);
    }
    stopwatchTick = requestAnimationFrame(tick);
  }

  function stopStopwatch() {
    if (stopwatchTick) {
      cancelAnimationFrame(stopwatchTick);
      stopwatchTick = null;
    }
    stopwatchStart = null;
    document.getElementById('stopwatch').classList.remove('running');
  }

  function resetStopwatch() {
    stopStopwatch();
    document.getElementById('stopwatch').textContent = '0:00.0';
  }

  function float32ToInt16Pcm(float32Array) {
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
      const s = Math.max(-1, Math.min(1, float32Array[i]));
      int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return int16;
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
    return btoa(binary);
  }

  function base64ToArrayBuffer(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return bytes.buffer;
  }

  function makeWavBlob(pcmInt16, sampleRate) {
    const numChannels = 1;
    const sampleWidth = 2;
    const byteRate = sampleRate * numChannels * sampleWidth;
    const dataSize = pcmInt16.length * sampleWidth;
    const buffer = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buffer);
    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeStr(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeStr(8, 'WAVE');
    writeStr(12, 'fmt ');
    view.setUint32(16, 16, true); // chunk size
    view.setUint16(20, 1, true);  // PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, numChannels * sampleWidth, true);
    view.setUint16(34, 16, true); // bits per sample
    writeStr(36, 'data');
    view.setUint32(40, dataSize, true);
    const view16 = new DataView(buffer, 44);
    new Int16Array(pcmInt16).forEach((v, i) => {
      view16.setInt16(i * 2, v, true);
    });
    return new Blob([buffer], { type: 'audio/wav' });
  }

  function resampleTo16k(audioBuffer) {
    const srcRate = audioBuffer.sampleRate;
    const srcLength = audioBuffer.length;
    const dstLength = Math.round((srcLength / srcRate) * PCM_SAMPLE_RATE);
    const src = audioBuffer.getChannelData(0);
    const dst = new Float32Array(dstLength);
    for (let i = 0; i < dstLength; i++) {
      const srcI = (i / dstLength) * srcLength;
      const j = Math.floor(srcI);
      const f = srcI - j;
      dst[i] = j + 1 < srcLength ? src[j] * (1 - f) + src[j + 1] * f : src[j];
    }
    return dst;
  }

  function resamplePcmInt16(srcRate, dstRate, pcmInt16) {
    const srcLength = pcmInt16.length;
    const dstLength = Math.round((srcLength / srcRate) * dstRate);
    const dst = new Int16Array(dstLength);
    for (let i = 0; i < dstLength; i++) {
      const srcI = (i / dstLength) * srcLength;
      const j = Math.floor(srcI);
      const f = srcI - j;
      const a = pcmInt16[j];
      const b = j + 1 < srcLength ? pcmInt16[j + 1] : a;
      const v = a * (1 - f) + b * f;
      dst[i] = Math.max(-32768, Math.min(32767, Math.round(v)));
    }
    return dst;
  }

  // —— Get state (REST) ——
  document.getElementById('getState').addEventListener('click', async function () {
    const base = getApiBase();
    const jwt = getJwt();
    if (!base || !jwt) {
      setStateDisplay('Set API base URL and JWT.');
      return;
    }
    setStateDisplay('Loading…');
    try {
      const r = await fetch(base + '/state', {
        headers: { Authorization: 'Bearer ' + jwt },
      });
      const data = await r.json().catch(() => ({}));
      if (!r.ok) {
        setStateDisplay('Error ' + r.status + ': ' + (data.detail || r.statusText));
        return;
      }
      setStateDisplay(JSON.stringify(data, null, 2));
    } catch (e) {
      setStateDisplay('Request failed: ' + e.message);
    }
  });

  // —— Train from mic (WebSocket) ——
  let trainWs = null;
  let trainStream = null;
  let trainContext = null;
  let trainProcessor = null;
  let trainChunkBuffer = [];
  let trainChunkInterval = null;
  let trainStopping = false;

  document.getElementById('trainStop').addEventListener('click', function () {
    if (!trainWs || trainWs.readyState !== WebSocket.OPEN) return;
    trainStopping = true; // used in ws.onmessage when train_result arrives
    setTrainStatus('Stopping and training…');
    trainWs.send(JSON.stringify({ operation: 'end' }));
    // Cleanup happens in onmessage when train_result is received
  });

  document.getElementById('trainStart').addEventListener('click', async function () {
    const base = getApiBase();
    const jwt = getJwt();
    if (!base || !jwt) {
      setTrainStatus('Set API base URL and JWT.', true);
      return;
    }
    const wsBase = base.replace(/^http/, 'ws');
    const wsUrl = wsBase + '/train?token=' + encodeURIComponent(jwt);

    trainStopping = false;
    resetStopwatch();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      trainStream = stream;

      const context = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: PCM_SAMPLE_RATE,
      });
      trainContext = context;

      const source = context.createMediaStreamSource(stream);
      const bufferSize = 2048;
      const processor = context.createScriptProcessor(bufferSize, 1, 1);
      trainProcessor = processor;

      const ws = new WebSocket(wsUrl);
      trainWs = ws;

      ws.onopen = function () {
        setTrainStatus('Connected. Recording… (up to 20s)');
        document.getElementById('trainStart').disabled = true;
        document.getElementById('trainStop').disabled = false;
        startStopwatch();
        ws.send(JSON.stringify({ operation: 'start' }));
      };

      ws.onmessage = function (event) {
        const msg = JSON.parse(event.data);
        if (msg.type === 'started') {
          setTrainStatus('Started. Recording… (up to 20s)');
        } else if (msg.type === 'chunk_received') {
          // no status change
        } else if (msg.type === 'train_result') {
          stopStopwatch();
          setTrainStatus('Trained. voice_id: ' + (msg.trained_voice_id || ''));
          if (trainStopping) {
            trainWs.close();
            trainWs = null;
            trainStopping = false;
            if (trainChunkInterval) clearInterval(trainChunkInterval);
            trainChunkInterval = null;
            if (trainProcessor && trainContext) {
              try { trainProcessor.disconnect(); } catch (_) {}
            }
            if (trainStream) {
              trainStream.getTracks().forEach((t) => t.stop());
              trainStream = null;
            }
            document.getElementById('trainStart').disabled = false;
            document.getElementById('trainStop').disabled = true;
          }
        } else if (msg.error) {
          setTrainStatus('Error: ' + msg.error, true);
        }
      };

      ws.onerror = function () {
        setTrainStatus('WebSocket error.', true);
      };

      ws.onclose = function () {
        setTrainStatus('WebSocket closed.');
      };

      trainChunkBuffer = [];
      processor.onaudioprocess = function (e) {
        const input = e.inputBuffer.getChannelData(0);
        const int16 = float32ToInt16Pcm(input);
        trainChunkBuffer.push(int16);
      };

      source.connect(processor);
      processor.connect(context.destination); // optional: hear yourself; remove to avoid feedback

      trainChunkInterval = setInterval(function () {
        if (trainChunkBuffer.length === 0 || ws.readyState !== WebSocket.OPEN) return;
        const totalLen = trainChunkBuffer.reduce((acc, a) => acc + a.length, 0);
        const combined = new Int16Array(totalLen);
        let offset = 0;
        for (const a of trainChunkBuffer) {
          combined.set(a, offset);
          offset += a.length;
        }
        trainChunkBuffer = [];
        const b64 = arrayBufferToBase64(combined.buffer);
        ws.send(JSON.stringify({ reference: b64 }));
      }, TRAIN_CHUNK_MS);
    } catch (e) {
      setTrainStatus('Error: ' + e.message, true);
      if (trainStream) trainStream.getTracks().forEach((t) => t.stop());
      trainStream = null;
    }
  });

  // —— TTS & Clone (WebSocket) ——
  document.getElementById('ttsClone').addEventListener('click', async function () {
    const base = getApiBase();
    const jwt = getJwt();
    const fileInput = document.getElementById('cloneFile');
    const file = fileInput && fileInput.files && fileInput.files[0];
    const text = (document.getElementById('ttsText').value || '').trim();
    if (!base || !jwt) {
      setCloneStatus('Set API base URL and JWT.', true);
      return;
    }
    if (!file && !text) {
      setCloneStatus('Upload an audio file or enter TTS text.', true);
      return;
    }

    let pcmBase64 = null;

    try {
      if (file) {
        setCloneStatus('Reading uploaded file…');
        try {
          const arrayBuffer = await file.arrayBuffer();
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
          const float32 = resampleTo16k(decoded);
          const int16 = float32ToInt16Pcm(float32);
          pcmBase64 = arrayBufferToBase64(int16.buffer);
        } catch (e) {
          setCloneStatus('Could not decode audio file: ' + e.message, true);
          return;
        }
      } else {
        if (typeof puter === 'undefined' || !puter.ai || !puter.ai.txt2speech) {
          setCloneStatus('Puter.js failed to load. Check the script or upload an audio file.', true);
          return;
        }
        setCloneStatus('Generating TTS with Puter.js…');
        const audio = await puter.ai.txt2speech(text, { language: 'en-US' }).catch((e) => {
          setCloneStatus('Puter TTS failed: ' + (e && e.message ? e.message : String(e)), true);
          return null;
        });
        if (!audio) return;
        const src = audio.src || audio.currentSrc;
        if (!src) {
          setCloneStatus('TTS did not return playable audio.', true);
          return;
        }
        await new Promise((resolve, reject) => {
          if (audio.readyState >= 2) resolve();
          else {
            audio.oncanplaythrough = () => resolve();
            audio.onerror = () => reject(new Error('TTS audio failed to load'));
          }
        });
        const arrayBuffer = await fetch(src).then((r) => r.arrayBuffer()).catch((e) => {
          setCloneStatus('Could not read TTS audio: ' + e.message, true);
          return null;
        });
        if (!arrayBuffer) return;
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0)).catch((e) => {
          setCloneStatus('Could not decode TTS audio: ' + e.message, true);
          return null;
        });
        if (!decoded) return;
        const float32 = resampleTo16k(decoded);
        const int16 = float32ToInt16Pcm(float32);
        pcmBase64 = arrayBufferToBase64(int16.buffer);
      }

      if (!pcmBase64) {
        setCloneStatus('No audio to clone. Upload a file or share tab with audio.', true);
        return;
      }

      setCloneStatus('Cloning…');
      const wsBase = base.replace(/^http/, 'ws');
      const wsUrl = wsBase + '/clone?token=' + encodeURIComponent(jwt);
      const ws = new WebSocket(wsUrl);

      const result = await new Promise((resolve, reject) => {
        ws.onopen = function () {
          ws.send(JSON.stringify({ base: pcmBase64 }));
        };
        ws.onmessage = function (event) {
          const msg = JSON.parse(event.data);
          if (msg.type === 'clone_result' && msg.output_path) {
            resolve(msg.output_path);
          } else if (msg.error) {
            reject(new Error(msg.error));
          }
        };
        ws.onerror = () => reject(new Error('WebSocket error'));
        ws.onclose = () => reject(new Error('WebSocket closed before result'));
      });

      ws.close();

      const pcmBytes = base64ToArrayBuffer(result);
      const pcmView = new DataView(pcmBytes);
      const numSamples = pcmView.byteLength / 2;
      const pcm16k = new Int16Array(numSamples);
      for (let i = 0; i < numSamples; i++) pcm16k[i] = pcmView.getInt16(i * 2, true);
      const pcmInt16 = resamplePcmInt16(CLONE_OUTPUT_SAMPLE_RATE, PCM_SAMPLE_RATE, pcm16k);
      const wavBlob = makeWavBlob(pcmInt16, PCM_SAMPLE_RATE);
      const url = URL.createObjectURL(wavBlob);
      const audioEl = document.getElementById('cloneAudio');
      audioEl.src = url;
      audioEl.play().catch(() => {});
      setCloneStatus('Done. Playing cloned audio.');
    } catch (e) {
      setCloneStatus('Error: ' + e.message, true);
    }
  });

  // —— Play original TTS (no clone) ——
  document.getElementById('playOriginalTts').addEventListener('click', async function () {
    const text = (document.getElementById('ttsText').value || '').trim();
    if (!text) {
      setCloneStatus('Enter TTS text first.', true);
      return;
    }
    if (typeof puter === 'undefined' || !puter.ai || !puter.ai.txt2speech) {
      setCloneStatus('Puter.js failed to load.', true);
      return;
    }
    setCloneStatus('Generating original TTS…');
    try {
      const audio = await puter.ai.txt2speech(text, { language: 'en-US' });
      const el = document.getElementById('originalTtsAudio');
      el.src = audio.src || audio.currentSrc || '';
      el.play().catch(() => {});
      setCloneStatus('Playing original TTS.');
    } catch (e) {
      setCloneStatus('Original TTS failed: ' + (e && e.message ? e.message : String(e)), true);
    }
  });
})();
