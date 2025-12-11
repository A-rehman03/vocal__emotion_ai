import React, { useRef, useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Mic,
  MicOff,
  Upload,
  Brain,
  AlertCircle,
  CheckCircle,
  Play,
  Pause,
  RotateCcw,
  Sparkles,
  AlertTriangle
} from 'lucide-react';
import ModernLoader from '../components/ui/ModernLoader';
import EmotionDisplay, { EmotionBreakdown } from '../components/ui/EmotionDisplay';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

export default function EmotionAnalysis() {
  const fileInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioRef = useRef(null);
  const [recording, setRecording] = useState(false);
  const [chunks, setChunks] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [recordingTime, setRecordingTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);

  useEffect(() => {
    let interval;
    if (recording) {
      interval = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [recording]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleUpload = async (file) => {
    setError('');
    setLoading(true);
    setResult(null);
    setAudioUrl(URL.createObjectURL(file));

    try {
      // Validate file size
      if (file.size > 10 * 1024 * 1024) { // 10MB
        throw new Error('File too large. Maximum size is 10MB');
      }

      // Validate file type
      const allowedTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/m4a', 'audio/webm', 'audio/ogg'];
      if (!allowedTypes.includes(file.type) && !file.name.match(/\.(wav|mp3|m4a|webm|ogg)$/i)) {
        throw new Error('Unsupported file format. Please use WAV, MP3, M4A, WebM, or OGG files');
      }

      const formData = new FormData();
      formData.append('audio', file);

      const token = localStorage.getItem('token');
      const headers = token ? { 'Authorization': `Bearer ${token}` } : {};

      const res = await fetch('/api/emotion/analyze', {
        method: 'POST',
        headers: headers,
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.message || data.error || 'Prediction failed');
      }

      // Adapt Node backend response to expected format
      let finalResult = data;
      if (data.analysis && !data.labels) {
        const emotion = data.analysis.detectedEmotion.toLowerCase();
        const confidence = parseFloat(data.analysis.confidence || 0.8);

        // Construct probabilities
        const emotions = ['happy', 'sad', 'angry', 'neutral', 'excited', 'calm'];
        const probs = {};
        const remainingProb = (1 - confidence) / (emotions.length - 1);

        emotions.forEach(e => {
          probs[e] = e.toLowerCase() === emotion ? confidence : remainingProb;
        });

        finalResult = {
          labels: {
            top1: emotion,
            probs: probs
          },
          audio_info: data.audioInfo || {
            duration: 0,
            sample_rate: 16000,
            channels: 1
          },
          input_shape: [1, 16000], // Mock data for UI
          features: { sequence_shape: [1, 128, 128] } // Mock data for UI
        };
      }

      // Validate response
      if (!finalResult.labels || !finalResult.labels.top1) {
        throw new Error('Invalid response from server');
      }

      setResult(finalResult);
    } catch (e) {
      console.error('Upload error:', e);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const onFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleUpload(file);
  };

  const startRecording = async () => {
    setError('');
    setRecordingTime(0);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
          channelCount: 1
        }
      });

      // Try different MIME types in order of preference
      const mimeTypes = [
        'audio/webm;codecs=opus',
        'audio/webm',
        'audio/mp4;codecs=mp4a.40.2',
        'audio/mp4',
        'audio/wav',
        'audio/ogg;codecs=opus'
      ];

      let mimeType = mimeTypes.find(type => MediaRecorder.isTypeSupported(type)) || 'audio/webm';
      console.log('Using MIME type:', mimeType);

      const mr = new MediaRecorder(stream, {
        mimeType,
        audioBitsPerSecond: 128000
      });
      setChunks([]);

      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          setChunks((prev) => [...prev, e.data]);
        }
      };

      mr.onstop = async () => {
        try {
          const blob = new Blob(chunks, { type: mr.mimeType });
          console.log('Recorded blob:', blob.type, blob.size, 'bytes');

          // Always try to convert to WAV for better backend compatibility
          try {
            const wavBlob = await convertToWav(blob);
            const file = new File([wavBlob], 'recording.wav', { type: 'audio/wav' });
            console.log('Converted to WAV:', wavBlob.type, wavBlob.size, 'bytes');
            setAudioUrl(URL.createObjectURL(wavBlob));
            handleUpload(file);
          } catch (conversionError) {
            console.warn('WAV conversion failed, using original format:', conversionError);
            // Fallback to original blob
            const file = new File([blob], `recording.${mimeType.split('/')[1].split(';')[0]}`, { type: mr.mimeType });
            setAudioUrl(URL.createObjectURL(blob));
            handleUpload(file);
          }
        } catch (error) {
          console.error('Error processing recorded audio:', error);
          setError('Failed to process recorded audio. Please try again.');
        }
      };

      mediaRecorderRef.current = mr;
      mr.start(100); // Collect data every 100ms
      setRecording(true);
    } catch (e) {
      console.error('Recording failed:', e);
      setError(`Recording failed: ${e.message}`);
    }
  };

  // Convert audio blob to WAV format
  const convertToWav = async (blob) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const arrayBuffer = reader.result;
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();

        audioContext.decodeAudioData(arrayBuffer)
          .then(audioBuffer => {
            const wavBlob = audioBufferToWav(audioBuffer);
            resolve(wavBlob);
          })
          .catch(reject);
      };
      reader.onerror = reject;
      reader.readAsArrayBuffer(blob);
    });
  };

  // Convert AudioBuffer to WAV blob
  const audioBufferToWav = (audioBuffer) => {
    const length = audioBuffer.length;
    const sampleRate = audioBuffer.sampleRate;
    const numberOfChannels = audioBuffer.numberOfChannels;
    const arrayBuffer = new ArrayBuffer(44 + length * numberOfChannels * 2);
    const view = new DataView(arrayBuffer);

    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * numberOfChannels * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numberOfChannels * 2, true);
    view.setUint16(32, numberOfChannels * 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * numberOfChannels * 2, true);

    // Convert audio data
    let offset = 44;
    for (let i = 0; i < length; i++) {
      for (let channel = 0; channel < numberOfChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, audioBuffer.getChannelData(channel)[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
  };

  const stopRecording = () => {
    const mr = mediaRecorderRef.current;
    if (mr && mr.state !== 'inactive') {
      mr.stop();
      setRecording(false);
    }
  };

  const togglePlayback = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const resetAnalysis = () => {
    setResult(null);
    setError('');
    setAudioUrl(null);
    setIsPlaying(false);
    setRecordingTime(0);
  };

  const getTopEmotion = () => {
    if (!result?.labels?.probs) return null;
    const emotions = Object.entries(result.labels.probs);
    return emotions.sort((a, b) => b[1] - a[1])[0];
  };

  const getConfidenceLevel = (confidence) => {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    if (confidence >= 0.3) return 'low';
    return 'very-low';
  };

  const topEmotion = getTopEmotion();

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/30 dark:from-dark-900 dark:via-dark-800 dark:to-dark-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <motion.div
            className="inline-flex items-center space-x-3 bg-white/80 dark:bg-dark-800/80 backdrop-blur-xl text-primary-700 dark:text-primary-300 px-6 py-3 rounded-full text-sm font-semibold mb-8 border border-white/20 dark:border-dark-700/50 shadow-xl"
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.4, delay: 0.2 }}
          >
            <Brain className="h-5 w-5" />
            <span>AI-Powered Emotion Recognition</span>
            <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse"></div>
          </motion.div>
          <h1 className="text-5xl md:text-6xl font-bold text-gray-900 dark:text-white mb-6">
            Voice Emotion
            <span className="gradient-text block">Analysis</span>
          </h1>
          <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
            Upload an audio file or record your voice to discover the emotions hidden in your speech patterns
          </p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          {/* Input Section */}
          <motion.div
            className="space-y-8"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            {/* Upload/Record Controls */}
            <div className="card-gradient">
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center">
                  <div className="h-12 w-12 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-2xl flex items-center justify-center mr-4 shadow-lg">
                    <Mic className="h-6 w-6 text-white" />
                  </div>
                  Audio Input
                </h2>
                <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full opacity-20"></div>
              </div>

              <div className="space-y-6">
                {/* File Upload */}
                <div className="relative">
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="w-full p-8 border-2 border-dashed border-primary-300 dark:border-primary-600 rounded-2xl hover:border-primary-500 dark:hover:border-primary-400 hover:bg-primary-50/50 dark:hover:bg-primary-900/20 transition-all duration-300 group hover-lift"
                    disabled={loading || recording}
                  >
                    <div className="flex flex-col items-center space-y-4">
                      <div className="h-16 w-16 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300 shadow-lg">
                        <Upload className="h-8 w-8 text-white" />
                      </div>
                      <div className="text-center">
                        <p className="text-xl font-bold text-gray-700 dark:text-gray-300 group-hover:text-primary-700 dark:group-hover:text-primary-300 mb-2">
                          Upload Audio File
                        </p>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                          WAV, MP3, M4A, or WebM format
                        </p>
                      </div>
                    </div>
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="audio/*"
                    className="hidden"
                    onChange={onFileChange}
                    disabled={loading || recording}
                  />
                </div>

                {/* Recording Controls */}
                <div className="flex items-center justify-center space-x-4">
                  <div className="text-center">
                    <p className="text-sm font-semibold text-gray-500 dark:text-gray-400 mb-2">OR</p>
                  </div>
                </div>

                <div className="flex flex-col items-center space-y-6">
                  {!recording ? (
                    <button
                      onClick={startRecording}
                      disabled={loading}
                      className="btn-primary px-10 py-4 text-lg font-bold flex items-center space-x-3 disabled:opacity-50 disabled:cursor-not-allowed group hover-lift hover-glow"
                    >
                      <Mic className="h-6 w-6 group-hover:scale-110 transition-transform" />
                      <span>Start Recording</span>
                    </button>
                  ) : (
                    <div className="flex flex-col items-center space-y-6">
                      <div className="flex items-center space-x-4">
                        <div className="flex items-center space-x-3 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300 px-6 py-3 rounded-2xl border border-red-200 dark:border-red-800">
                          <div className="w-4 h-4 bg-red-500 rounded-full animate-pulse"></div>
                          <span className="font-mono text-xl font-bold">{formatTime(recordingTime)}</span>
                        </div>
                      </div>
                      <button
                        onClick={stopRecording}
                        className="btn-outline px-10 py-4 text-lg font-bold flex items-center space-x-3 border-red-300 dark:border-red-600 text-red-700 dark:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/20 hover-lift"
                      >
                        <MicOff className="h-6 w-6" />
                        <span>Stop Recording</span>
                      </button>
                    </div>
                  )}
                </div>

                {/* Audio Playback */}
                {audioUrl && (
                  <motion.div
                    className="bg-white/80 dark:bg-dark-800/80 backdrop-blur-xl rounded-2xl p-6 border border-white/20 dark:border-dark-700/50 shadow-xl"
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.4 }}
                  >
                    <div className="flex items-center justify-between mb-4">
                      <h3 className="text-lg font-bold text-gray-700 dark:text-gray-300">Audio Preview</h3>
                      <button
                        onClick={togglePlayback}
                        className="p-3 rounded-xl bg-gradient-to-r from-primary-500 to-secondary-500 hover:from-primary-600 hover:to-secondary-600 text-white transition-all duration-300 hover:scale-110 shadow-lg"
                      >
                        {isPlaying ? (
                          <Pause className="h-5 w-5" />
                        ) : (
                          <Play className="h-5 w-5" />
                        )}
                      </button>
                    </div>
                    <audio
                      ref={audioRef}
                      src={audioUrl}
                      onEnded={() => setIsPlaying(false)}
                      className="w-full"
                    />
                  </motion.div>
                )}
              </div>
            </div>

            {/* Loading State */}
            {loading && (
              <motion.div
                className="card-gradient text-center"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4 }}
              >
                <ModernLoader
                  size="xl"
                  text="Analyzing Audio"
                  showBrain={true}
                  variant="brain"
                />
                <div className="mt-6">
                  <p className="text-lg text-gray-600 dark:text-gray-300 font-medium">Our AI is processing your voice to detect emotions...</p>
                  <div className="mt-4 flex justify-center">
                    <div className="flex space-x-2">
                      <div className="w-3 h-3 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full animate-bounce"></div>
                      <div className="w-3 h-3 bg-gradient-to-r from-secondary-500 to-accent-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-3 h-3 bg-gradient-to-r from-accent-500 to-primary-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}

            {/* Error State */}
            {error && (
              <motion.div
                className="card-gradient border-2 border-red-200 dark:border-red-800 bg-red-50/50 dark:bg-red-900/20"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4 }}
              >
                <div className="flex items-start space-x-4">
                  <div className="h-12 w-12 bg-red-500 rounded-2xl flex items-center justify-center flex-shrink-0 shadow-lg">
                    <AlertCircle className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-red-800 dark:text-red-300 mb-2">Analysis Failed</h3>
                    <p className="text-red-700 dark:text-red-300 text-lg">{error}</p>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>

          {/* Results Section */}
          <motion.div
            className="space-y-8"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            {result && (
              <>
                {/* Main Result */}
                <motion.div
                  className="card-gradient"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5 }}
                >
                  <div className="flex items-center justify-between mb-8">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center">
                      <div className="h-12 w-12 bg-gradient-to-br from-green-500 to-accent-500 rounded-2xl flex items-center justify-center mr-4 shadow-lg">
                        <CheckCircle className="h-6 w-6 text-white" />
                      </div>
                      Analysis Results
                    </h2>
                    <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-accent-500 rounded-full opacity-20"></div>
                  </div>

                  {topEmotion && (
                    <div className="text-center mb-8">
                      <EmotionDisplay
                        emotion={topEmotion[0]}
                        confidence={topEmotion[1]}
                        size="xl"
                        animated={true}
                        className="animate-scale-in"
                      />
                      {result.labels?.top1 === 'uncertain' && (
                        <div className="mt-4 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
                          <div className="flex items-center justify-center space-x-2 text-amber-700 dark:text-amber-300">
                            <AlertTriangle className="h-4 w-4" />
                            <span className="text-sm font-medium">Low confidence detected</span>
                          </div>
                          <p className="text-xs text-amber-600 dark:text-amber-400 mt-1">
                            Try recording again with clearer speech
                          </p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Emotion Breakdown */}
                  {result.labels?.probs && (
                    <EmotionBreakdown
                      emotions={result.labels.probs}
                      className="animate-slide-in-right"
                    />
                  )}
                </motion.div>

                {/* Technical Details */}
                <motion.div
                  className="card-gradient"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <div className="flex items-center justify-between mb-8">
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white flex items-center">
                      <div className="h-10 w-10 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-2xl flex items-center justify-center mr-4 shadow-lg">
                        <Brain className="h-5 w-5 text-white" />
                      </div>
                      Technical Details
                    </h3>
                    <div className="w-6 h-6 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full opacity-20"></div>
                  </div>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center py-4 px-4 bg-white/50 dark:bg-dark-800/50 rounded-xl border border-white/20 dark:border-dark-700/50">
                      <span className="text-gray-600 dark:text-gray-300 font-semibold">Model Input Shape:</span>
                      <span className="font-mono text-white bg-gradient-to-r from-primary-500 to-secondary-500 px-3 py-1 rounded-lg text-sm font-bold">
                        {result.input_shape?.join(' × ') || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between items-center py-4 px-4 bg-white/50 dark:bg-dark-800/50 rounded-xl border border-white/20 dark:border-dark-700/50">
                      <span className="text-gray-600 dark:text-gray-300 font-semibold">Feature Shape:</span>
                      <span className="font-mono text-white bg-gradient-to-r from-secondary-500 to-accent-500 px-3 py-1 rounded-lg text-sm font-bold">
                        {result.features?.sequence_shape?.join(' × ') || 'N/A'}
                      </span>
                    </div>
                    {result.audio_info && (
                      <>
                        <div className="flex justify-between items-center py-4 px-4 bg-white/50 dark:bg-dark-800/50 rounded-xl border border-white/20 dark:border-dark-700/50">
                          <span className="text-gray-600 dark:text-gray-300 font-semibold">Duration:</span>
                          <span className="font-mono text-white bg-gradient-to-r from-accent-500 to-primary-500 px-3 py-1 rounded-lg text-sm font-bold">
                            {(result.audio_info.duration || 0).toFixed(2)}s
                          </span>
                        </div>
                        <div className="flex justify-between items-center py-4 px-4 bg-white/50 dark:bg-dark-800/50 rounded-xl border border-white/20 dark:border-dark-700/50">
                          <span className="text-gray-600 dark:text-gray-300 font-semibold">Sample Rate:</span>
                          <span className="font-mono text-white bg-gradient-to-r from-primary-500 to-secondary-500 px-3 py-1 rounded-lg text-sm font-bold">
                            {result.audio_info.sample_rate}Hz
                          </span>
                        </div>
                        <div className="flex justify-between items-center py-4 px-4 bg-white/50 dark:bg-dark-800/50 rounded-xl border border-white/20 dark:border-dark-700/50">
                          <span className="text-gray-600 dark:text-gray-300 font-semibold">Channels:</span>
                          <span className="font-mono text-white bg-gradient-to-r from-secondary-500 to-accent-500 px-3 py-1 rounded-lg text-sm font-bold">
                            {result.audio_info.channels}
                          </span>
                        </div>
                      </>
                    )}
                    <div className="flex justify-between items-center py-4 px-4 bg-white/50 dark:bg-dark-800/50 rounded-xl border border-white/20 dark:border-dark-700/50">
                      <span className="text-gray-600 dark:text-gray-300 font-semibold">Processing Time:</span>
                      <span className="font-mono text-white bg-gradient-to-r from-accent-500 to-primary-500 px-3 py-1 rounded-lg text-sm font-bold">
                        &lt; 1s
                      </span>
                    </div>
                    <div className="flex justify-between items-center py-4 px-4 bg-white/50 dark:bg-dark-800/50 rounded-xl border border-white/20 dark:border-dark-700/50">
                      <span className="text-gray-600 dark:text-gray-300 font-semibold">Model Status:</span>
                      <span className="flex items-center text-green-600 dark:text-green-400 font-bold">
                        <div className="w-3 h-3 bg-green-500 rounded-full mr-2 animate-pulse"></div>
                        Active
                      </span>
                    </div>
                  </div>
                </motion.div>

                {/* Reset Button */}
                <motion.div
                  className="flex justify-center"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.4 }}
                >
                  <button
                    onClick={resetAnalysis}
                    className="btn-outline px-10 py-4 flex items-center space-x-3 hover-lift group text-lg font-bold"
                  >
                    <RotateCcw className="h-6 w-6 group-hover:rotate-180 transition-transform duration-300" />
                    <span>Analyze Another Audio</span>
                  </button>
                </motion.div>
              </>
            )}

            {/* Empty State */}
            {!result && !loading && !error && (
              <motion.div
                className="card-gradient text-center py-16"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <div className="flex flex-col items-center space-y-8">
                  <div className="relative">
                    <div className="w-24 h-24 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-3xl flex items-center justify-center animate-float shadow-2xl">
                      <Brain className="h-12 w-12 text-white" />
                    </div>
                    <div className="absolute -top-3 -right-3 w-8 h-8 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full flex items-center justify-center animate-bounce shadow-lg">
                      <Sparkles className="h-4 w-4 text-white" />
                    </div>
                  </div>
                  <div className="space-y-4">
                    <h3 className="text-3xl font-bold text-gray-900 dark:text-white">Ready to Analyze</h3>
                    <p className="text-lg text-gray-600 dark:text-gray-300 max-w-md mx-auto leading-relaxed">
                      Upload an audio file or start recording to discover the emotions hidden in your voice
                    </p>
                  </div>
                  <div className="flex items-center space-x-6 text-sm text-gray-500 dark:text-gray-400">
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-gradient-to-r from-green-500 to-accent-500 rounded-full"></div>
                      <span className="font-semibold">AI-Powered</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full"></div>
                      <span className="font-semibold">Real-time</span>
                    </div>
                    <div className="flex items-center space-x-2">
                      <div className="w-3 h-3 bg-gradient-to-r from-secondary-500 to-accent-500 rounded-full"></div>
                      <span className="font-semibold">Accurate</span>
                    </div>
                  </div>
                </div>
              </motion.div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}