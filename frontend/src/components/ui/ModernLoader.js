import React from 'react';
import { Loader2, Brain, Sparkles } from 'lucide-react';

const ModernLoader = ({ 
  size = 'lg', 
  text = 'Loading...', 
  showBrain = false,
  variant = 'default' 
}) => {
  const sizeClasses = {
    sm: 'h-4 w-4',
    md: 'h-6 w-6',
    lg: 'h-8 w-8',
    xl: 'h-12 w-12'
  };

  const containerClasses = {
    sm: 'space-y-2',
    md: 'space-y-3',
    lg: 'space-y-4',
    xl: 'space-y-6'
  };

  if (variant === 'brain') {
    return (
      <div className={`flex flex-col items-center ${containerClasses[size]}`}>
        <div className="relative">
          <div className={`${sizeClasses[size]} text-primary-600 animate-spin`}>
            <Loader2 className="h-full w-full" />
          </div>
          {showBrain && (
            <div className="absolute inset-0 flex items-center justify-center">
              <Brain className="h-4 w-4 text-primary-400 animate-pulse" />
            </div>
          )}
        </div>
        <div className="text-center">
          <p className="text-sm font-medium text-gray-700">{text}</p>
          <div className="flex items-center justify-center space-x-1 mt-1">
            <div className="w-1 h-1 bg-primary-400 rounded-full animate-bounce"></div>
            <div className="w-1 h-1 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
            <div className="w-1 h-1 bg-primary-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
          </div>
        </div>
      </div>
    );
  }

  if (variant === 'sparkles') {
    return (
      <div className={`flex flex-col items-center ${containerClasses[size]}`}>
        <div className="relative">
          <div className={`${sizeClasses[size]} text-primary-600 animate-pulse`}>
            <Sparkles className="h-full w-full" />
          </div>
          <div className="absolute -top-1 -right-1 w-2 h-2 bg-yellow-400 rounded-full animate-ping"></div>
          <div className="absolute -bottom-1 -left-1 w-1 h-1 bg-blue-400 rounded-full animate-ping" style={{ animationDelay: '0.5s' }}></div>
        </div>
        <p className="text-sm font-medium text-gray-700">{text}</p>
      </div>
    );
  }

  return (
    <div className={`flex flex-col items-center ${containerClasses[size]}`}>
      <Loader2 className={`${sizeClasses[size]} text-primary-600 animate-spin`} />
      <p className="text-sm font-medium text-gray-700">{text}</p>
    </div>
  );
};

export default ModernLoader;
