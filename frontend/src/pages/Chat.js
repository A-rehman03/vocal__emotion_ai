import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import {
  Mic,
  Volume2,
  MessageCircle,
  Sparkles,
  Brain,
  Activity
} from 'lucide-react';
import { vapi, startAssistant, stopAssistant } from '../utils/vapi';

const Chat = () => {
  const { user } = useAuth();
  const [isConversationActive, setIsConversationActive] = useState(false);
  const [conversationState, setConversationState] = useState('idle'); // idle, listening, speaking
  const [currentEmotion, setCurrentEmotion] = useState(null);
  const [conversationCount, setConversationCount] = useState(0);
  const [volumeLevel, setVolumeLevel] = useState(0);
  const [recordingTime, setRecordingTime] = useState(0);
  const timerRef = useRef(null);

  useEffect(() => {
    const onCallStart = () => {
      setIsConversationActive(true);
      setConversationState('listening');
      setConversationCount(prev => prev + 1);
      setRecordingTime(0);
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    };

    const onCallEnd = () => {
      setIsConversationActive(false);
      setConversationState('idle');
      setCurrentEmotion(null);
      setVolumeLevel(0);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };

    const onSpeechStart = () => {
      setConversationState('speaking');
    };

    const onSpeechEnd = () => {
      setConversationState('listening');
    };

    const onVolumeLevel = (level) => {
      setVolumeLevel(level);
    };

    const onError = (e) => {
      console.error("Vapi error:", e);
      setIsConversationActive(false);
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };

    vapi.on("call-start", onCallStart);
    vapi.on("call-end", onCallEnd);
    vapi.on("speech-start", onSpeechStart);
    vapi.on("speech-end", onSpeechEnd);
    vapi.on("volume-level", onVolumeLevel);
    vapi.on("error", onError);

    return () => {
      vapi.off("call-start", onCallStart);
      vapi.off("call-end", onCallEnd);
      vapi.off("speech-start", onSpeechStart);
      vapi.off("speech-end", onSpeechEnd);
      vapi.off("volume-level", onVolumeLevel);
      vapi.off("error", onError);

      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, []);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleStartConversation = async () => {
    try {
      await startAssistant();
    } catch (e) {
      console.error("Failed to start assistant", e);
    }
  };

  const handleEndConversation = () => {
    stopAssistant();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/30 dark:from-dark-900 dark:via-dark-800 dark:to-dark-900 flex flex-col">
      {/* Header */}
      <motion.div
        className="bg-white/80 dark:bg-dark-800/80 backdrop-blur-xl border-b border-white/20 dark:border-dark-700/50 px-6 py-6 shadow-xl"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="h-12 w-12 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-2xl flex items-center justify-center shadow-lg">
              <Brain className="h-7 w-7 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Emotion AI Assistant</h1>
              <p className="text-sm text-gray-600 dark:text-gray-300 font-medium">Powered by Advanced Voice AI</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600 dark:text-gray-300 font-semibold">Online</span>
          </div>
        </div>
      </motion.div>

      {/* Conversation Area */}
      <div className="flex-1 flex items-center justify-center px-6 py-6">
        {!isConversationActive ? (
          <motion.div
            className="flex flex-col items-center justify-center text-center"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
          >
            <div className="relative mb-8">
              <div className="w-32 h-32 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-full flex items-center justify-center shadow-2xl">
                <MessageCircle className="h-16 w-16 text-white" />
              </div>
              <div className="absolute -top-2 -right-2 w-10 h-10 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full flex items-center justify-center animate-bounce shadow-lg">
                <Sparkles className="h-5 w-5 text-white" />
              </div>
            </div>
            <h3 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
              Ready to Chat?
            </h3>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-lg leading-relaxed mb-8">
              Start a natural voice conversation with our AI assistant.
            </p>
            <motion.button
              onClick={handleStartConversation}
              className="px-8 py-4 bg-gradient-to-r from-primary-500 to-secondary-500 hover:from-primary-600 hover:to-secondary-600 text-white text-lg font-semibold rounded-2xl shadow-2xl transition-all duration-300 hover:scale-105"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Start Conversation
            </motion.button>
          </motion.div>
        ) : (
          <motion.div
            className="flex flex-col items-center justify-center text-center"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6 }}
          >
            {/* Main Conversation Icon */}
            <div className="relative mb-8">
              <motion.div
                className={`w-32 h-32 rounded-full flex items-center justify-center shadow-2xl ${conversationState === 'listening' ? 'bg-gradient-to-br from-red-500 to-red-600' :
                  conversationState === 'speaking' ? 'bg-gradient-to-br from-green-500 to-emerald-500' :
                    'bg-gradient-to-br from-primary-500 to-secondary-500'
                  }`}
                animate={{
                  scale: conversationState === 'listening' || conversationState === 'speaking' ? [1, 1.05 + volumeLevel, 1] : 1,
                }}
                transition={{
                  duration: 0.2,
                  ease: "easeInOut"
                }}
              >
                {conversationState === 'listening' ? (
                  <Mic className="h-16 w-16 text-white" />
                ) : conversationState === 'speaking' ? (
                  <Volume2 className="h-16 w-16 text-white" />
                ) : (
                  <Activity className="h-16 w-16 text-white" />
                )}
              </motion.div>

              {/* Audio Level Visualization */}
              {conversationState === 'listening' && (
                <div className="absolute inset-0 rounded-full border-4 border-red-300 animate-ping" style={{ opacity: volumeLevel * 2 }}></div>
              )}

              {/* Emotion Indicator - Hidden for now as Vapi doesn't provide it directly */}
              {currentEmotion && (
                <div className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 px-4 py-2 bg-gradient-to-r from-primary-500 to-secondary-500 text-white text-sm font-semibold rounded-full shadow-lg">
                  {currentEmotion}
                </div>
              )}
            </div>

            {/* Conversation Status */}
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              {conversationState === 'listening' ? 'Listening...' :
                conversationState === 'speaking' ? 'Assistant is speaking...' :
                  'Conversation Active'}
            </h3>

            <p className="text-lg text-gray-600 dark:text-gray-300 mb-6">
              {conversationState === 'listening' ? `Speak now... (${formatTime(recordingTime)})` :
                conversationState === 'speaking' ? 'AI is responding...' :
                  'Connecting...'}
            </p>

            {/* Conversation Stats */}
            <div className="flex items-center space-x-6 text-sm text-gray-500 dark:text-gray-400 mb-8">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-gradient-to-r from-green-500 to-accent-500 rounded-full"></div>
                <span className="font-semibold">Conversation #{conversationCount}</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full"></div>
                <span className="font-semibold">Voice AI</span>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex items-center space-x-4">
              <motion.button
                onClick={handleEndConversation}
                className="px-6 py-3 bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white font-semibold rounded-xl shadow-lg transition-all duration-300 hover:scale-105"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                End Conversation
              </motion.button>
            </div>
          </motion.div>
        )}
      </div>

    </div>
  );
};

export default Chat;
