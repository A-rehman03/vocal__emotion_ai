async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const source = audioCtx.createMediaStreamSource(stream);
  const processor = audioCtx.createScriptProcessor(4096, 1, 1);
  let audioData = [];
  processor.onaudioprocess = function (e) {
    audioData.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  };
  source.connect(processor);
  processor.connect(audioCtx.destination);
  return { stream, source, processor, audioData, audioCtx };
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function encodeWAV(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return view;
}

async function stopRecording(rec) {
  rec.source.disconnect();
  rec.processor.disconnect();
  rec.stream.getTracks().forEach((t) => t.stop());

  let totalLen = rec.audioData.reduce((acc, cur) => acc + cur.length, 0);
  const merged = new Float32Array(totalLen);
  let offset = 0;
  rec.audioData.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });

  const wavBuffer = encodeWAV(merged, rec.audioCtx.sampleRate);
  const blob = new Blob([wavBuffer], { type: "audio/wav" });
  return blob;
}

window.startRecording = startRecording;
window.stopRecording = stopRecording;
