import React from 'react';
import { 
  Smile, 
  Frown, 
  Meh, 
  Heart, 
  Zap, 
  AlertCircle, 
  TrendingUp, 
  Sparkles,
  Brain,
  HelpCircle
} from 'lucide-react';

const emotionIcons = {
  happy: Smile,
  sad: Frown,
  angry: Frown,
  neutral: Meh,
  excited: Zap,
  calm: Heart,
  anxious: AlertCircle,
  confident: TrendingUp,
  disgust: Frown,
  fear: AlertCircle,
  surprise: Sparkles,
  uncertain: HelpCircle
};

const emotionColors = {
  happy: {
    bg: 'bg-yellow-50 dark:bg-yellow-900/20',
    border: 'border-yellow-200 dark:border-yellow-800',
    text: 'text-yellow-800 dark:text-yellow-200',
    icon: 'text-yellow-600 dark:text-yellow-400',
    progress: 'bg-yellow-500'
  },
  sad: {
    bg: 'bg-blue-50 dark:bg-blue-900/20',
    border: 'border-blue-200 dark:border-blue-800',
    text: 'text-blue-800 dark:text-blue-200',
    icon: 'text-blue-600 dark:text-blue-400',
    progress: 'bg-blue-500'
  },
  angry: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    border: 'border-red-200 dark:border-red-800',
    text: 'text-red-800 dark:text-red-200',
    icon: 'text-red-600 dark:text-red-400',
    progress: 'bg-red-500'
  },
  neutral: {
    bg: 'bg-gray-50 dark:bg-gray-900/20',
    border: 'border-gray-200 dark:border-gray-800',
    text: 'text-gray-800 dark:text-gray-200',
    icon: 'text-gray-600 dark:text-gray-400',
    progress: 'bg-gray-500'
  },
  excited: {
    bg: 'bg-orange-50 dark:bg-orange-900/20',
    border: 'border-orange-200 dark:border-orange-800',
    text: 'text-orange-800 dark:text-orange-200',
    icon: 'text-orange-600 dark:text-orange-400',
    progress: 'bg-orange-500'
  },
  calm: {
    bg: 'bg-green-50 dark:bg-green-900/20',
    border: 'border-green-200 dark:border-green-800',
    text: 'text-green-800 dark:text-green-200',
    icon: 'text-green-600 dark:text-green-400',
    progress: 'bg-green-500'
  },
  anxious: {
    bg: 'bg-purple-50 dark:bg-purple-900/20',
    border: 'border-purple-200 dark:border-purple-800',
    text: 'text-purple-800 dark:text-purple-200',
    icon: 'text-purple-600 dark:text-purple-400',
    progress: 'bg-purple-500'
  },
  confident: {
    bg: 'bg-indigo-50 dark:bg-indigo-900/20',
    border: 'border-indigo-200 dark:border-indigo-800',
    text: 'text-indigo-800 dark:text-indigo-200',
    icon: 'text-indigo-600 dark:text-indigo-400',
    progress: 'bg-indigo-500'
  },
  disgust: {
    bg: 'bg-red-50 dark:bg-red-900/20',
    border: 'border-red-200 dark:border-red-800',
    text: 'text-red-800 dark:text-red-200',
    icon: 'text-red-600 dark:text-red-400',
    progress: 'bg-red-500'
  },
  fear: {
    bg: 'bg-purple-50 dark:bg-purple-900/20',
    border: 'border-purple-200 dark:border-purple-800',
    text: 'text-purple-800 dark:text-purple-200',
    icon: 'text-purple-600 dark:text-purple-400',
    progress: 'bg-purple-500'
  },
  surprise: {
    bg: 'bg-pink-50 dark:bg-pink-900/20',
    border: 'border-pink-200 dark:border-pink-800',
    text: 'text-pink-800 dark:text-pink-200',
    icon: 'text-pink-600 dark:text-pink-400',
    progress: 'bg-pink-500'
  },
  uncertain: {
    bg: 'bg-gray-50 dark:bg-gray-900/20',
    border: 'border-gray-200 dark:border-gray-800',
    text: 'text-gray-800 dark:text-gray-200',
    icon: 'text-gray-600 dark:text-gray-400',
    progress: 'bg-gray-500'
  }
};

const EmotionDisplay = ({ 
  emotion, 
  confidence, 
  size = 'lg',
  showConfidence = true,
  animated = true,
  className = '',
  note = null
}) => {
  const IconComponent = emotionIcons[emotion] || Brain;
  const colors = emotionColors[emotion] || emotionColors.neutral;
  
  const sizeClasses = {
    sm: {
      container: 'w-12 h-12',
      icon: 'h-6 w-6',
      text: 'text-sm',
      confidence: 'text-xs'
    },
    md: {
      container: 'w-16 h-16',
      icon: 'h-8 w-8',
      text: 'text-base',
      confidence: 'text-sm'
    },
    lg: {
      container: 'w-24 h-24',
      icon: 'h-12 w-12',
      text: 'text-2xl',
      confidence: 'text-lg'
    },
    xl: {
      container: 'w-32 h-32',
      icon: 'h-16 w-16',
      text: 'text-3xl',
      confidence: 'text-xl'
    }
  };

  const currentSize = sizeClasses[size];
  const isUncertain = emotion === 'uncertain' || confidence < 0.3;

  return (
    <div className={`flex flex-col items-center space-y-3 ${className}`}>
      <div className={`
        ${currentSize.container} 
        ${colors.bg} 
        ${colors.border} 
        border-2 
        rounded-full 
        flex items-center justify-center
        ${animated ? 'animate-scale-in hover-lift' : ''}
        transition-all duration-300
        ${isUncertain ? 'animate-pulse' : ''}
      `}>
        <IconComponent className={`${currentSize.icon} ${colors.icon}`} />
      </div>
      
      <div className="text-center">
        <h3 className={`${currentSize.text} font-bold ${colors.text} capitalize`}>
          {emotion === 'uncertain' ? 'Uncertain' : emotion}
        </h3>
        {showConfidence && (
          <p className={`${currentSize.confidence} font-semibold ${
            isUncertain ? 'text-orange-600 dark:text-orange-400' : 'text-primary-600 dark:text-primary-400'
          }`}>
            {Math.round(confidence * 100)}% Confidence
          </p>
        )}
        {note && (
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1 italic">
            {note}
          </p>
        )}
      </div>
    </div>
  );
};

export const EmotionBreakdown = ({ emotions, className = '' }) => {
  const sortedEmotions = Object.entries(emotions || {})
    .sort((a, b) => b[1] - a[1]);

  return (
    <div className={`space-y-3 ${className}`}>
      <h4 className="font-semibold text-gray-900 dark:text-white mb-4">Emotion Breakdown</h4>
      {sortedEmotions.map(([emotion, confidence]) => {
        const IconComponent = emotionIcons[emotion] || Brain;
        const colors = emotionColors[emotion] || emotionColors.neutral;
        
        return (
          <div key={emotion} className="flex items-center space-x-3 animate-fade-in-up">
            <div className="flex items-center space-x-2 min-w-0 flex-1">
              <IconComponent className={`h-5 w-5 ${colors.icon}`} />
              <span className="font-medium text-gray-700 dark:text-gray-300 capitalize">{emotion}</span>
            </div>
            <div className="flex-1 bg-gray-200 dark:bg-dark-700 rounded-full h-2 overflow-hidden">
              <div
                className={`h-2 rounded-full ${colors.progress} transition-all duration-1000 ease-out`}
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
            <span className="text-sm font-semibold text-gray-600 dark:text-gray-400 min-w-0">
              {Math.round(confidence * 100)}%
            </span>
          </div>
        );
      })}
    </div>
  );
};

export default EmotionDisplay;