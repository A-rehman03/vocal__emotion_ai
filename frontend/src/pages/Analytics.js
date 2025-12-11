import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart3, 
  TrendingUp, 
  Calendar, 
  Clock, 
  Brain, 
  MessageCircle,
  Mic,
  Activity,
  Download,
  Filter,
  RefreshCw
} from 'lucide-react';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const Analytics = () => {
  const [isLoading, setIsLoading] = useState(true);
  const [timeRange, setTimeRange] = useState('7d');
  const [analytics, setAnalytics] = useState(null);

  useEffect(() => {
    // Simulate loading analytics data
    const loadAnalytics = async () => {
      setIsLoading(true);
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setAnalytics({
        totalSessions: 24,
        totalEmotions: 156,
        averageAccuracy: 94.2,
        totalTime: 142, // minutes
        emotionsBreakdown: {
          happy: 35,
          neutral: 28,
          sad: 18,
          angry: 12,
          excited: 15,
          calm: 20,
          anxious: 8,
          confident: 20
        },
        weeklyData: [
          { day: 'Mon', sessions: 3, emotions: 12, accuracy: 92 },
          { day: 'Tue', sessions: 5, emotions: 18, accuracy: 95 },
          { day: 'Wed', sessions: 2, emotions: 8, accuracy: 88 },
          { day: 'Thu', sessions: 4, emotions: 15, accuracy: 96 },
          { day: 'Fri', sessions: 6, emotions: 22, accuracy: 94 },
          { day: 'Sat', sessions: 3, emotions: 11, accuracy: 91 },
          { day: 'Sun', sessions: 1, emotions: 4, accuracy: 89 }
        ],
        recentActivity: [
          {
            id: 1,
            type: 'emotion',
            emotion: 'happy',
            confidence: 92,
            timestamp: new Date(Date.now() - 1000 * 60 * 15),
            duration: 45
          },
          {
            id: 2,
            type: 'chat',
            message: 'Started conversation about work stress',
            timestamp: new Date(Date.now() - 1000 * 60 * 30),
            duration: 120
          },
          {
            id: 3,
            type: 'emotion',
            emotion: 'calm',
            confidence: 87,
            timestamp: new Date(Date.now() - 1000 * 60 * 45),
            duration: 30
          },
          {
            id: 4,
            type: 'emotion',
            emotion: 'excited',
            confidence: 89,
            timestamp: new Date(Date.now() - 1000 * 60 * 60),
            duration: 25
          }
        ]
      });
      setIsLoading(false);
    };

    loadAnalytics();
  }, [timeRange]);

  const formatTime = (minutes) => {
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return hours > 0 ? `${hours}h ${mins}m` : `${mins}m`;
  };

  const getEmotionColor = (emotion) => {
    const colors = {
      happy: 'text-yellow-500',
      sad: 'text-blue-500',
      angry: 'text-red-500',
      neutral: 'text-gray-500',
      excited: 'text-orange-500',
      calm: 'text-green-500',
      anxious: 'text-purple-500',
      confident: 'text-indigo-500'
    };
    return colors[emotion] || 'text-gray-500';
  };

  const getEmotionBgColor = (emotion) => {
    const colors = {
      happy: 'bg-yellow-100 dark:bg-yellow-900/20',
      sad: 'bg-blue-100 dark:bg-blue-900/20',
      angry: 'bg-red-100 dark:bg-red-900/20',
      neutral: 'bg-gray-100 dark:bg-gray-900/20',
      excited: 'bg-orange-100 dark:bg-orange-900/20',
      calm: 'bg-green-100 dark:bg-green-900/20',
      anxious: 'bg-purple-100 dark:bg-purple-900/20',
      confident: 'bg-indigo-100 dark:bg-indigo-900/20'
    };
    return colors[emotion] || 'bg-gray-100 dark:bg-gray-900/20';
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-dark-900 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-gray-600 dark:text-gray-300">Loading analytics...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-dark-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">Analytics Dashboard</h1>
              <p className="text-gray-600 dark:text-gray-300">Track your emotional journey and AI interactions</p>
            </div>
            <div className="flex items-center space-x-3">
              <select
                value={timeRange}
                onChange={(e) => setTimeRange(e.target.value)}
                className="input-field w-32"
              >
                <option value="7d">Last 7 days</option>
                <option value="30d">Last 30 days</option>
                <option value="90d">Last 90 days</option>
              </select>
              <button className="btn-outline flex items-center space-x-2">
                <Download className="h-4 w-4" />
                <span>Export</span>
              </button>
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Total Sessions</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{analytics.totalSessions}</p>
                <p className="text-sm text-green-600 dark:text-green-400">+12% from last week</p>
              </div>
              <div className="h-12 w-12 bg-primary-100 dark:bg-primary-900/30 rounded-lg flex items-center justify-center">
                <MessageCircle className="h-6 w-6 text-primary-600 dark:text-primary-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="card"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Emotions Analyzed</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{analytics.totalEmotions}</p>
                <p className="text-sm text-green-600 dark:text-green-400">+8% from last week</p>
              </div>
              <div className="h-12 w-12 bg-secondary-100 dark:bg-secondary-900/30 rounded-lg flex items-center justify-center">
                <Brain className="h-6 w-6 text-secondary-600 dark:text-secondary-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Accuracy Rate</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{analytics.averageAccuracy}%</p>
                <p className="text-sm text-green-600 dark:text-green-400">+2.1% from last week</p>
              </div>
              <div className="h-12 w-12 bg-green-100 dark:bg-green-900/30 rounded-lg flex items-center justify-center">
                <TrendingUp className="h-6 w-6 text-green-600 dark:text-green-400" />
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="card"
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-600 dark:text-gray-300 mb-1">Total Time</p>
                <p className="text-2xl font-bold text-gray-900 dark:text-white">{formatTime(analytics.totalTime)}</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">This week</p>
              </div>
              <div className="h-12 w-12 bg-orange-100 dark:bg-orange-900/30 rounded-lg flex items-center justify-center">
                <Clock className="h-6 w-6 text-orange-600 dark:text-orange-400" />
              </div>
            </div>
          </motion.div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
          {/* Weekly Activity Chart */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="card"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Weekly Activity</h3>
            <div className="space-y-4">
              {analytics.weeklyData.map((day, index) => (
                <div key={day.day} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="w-8 text-sm font-medium text-gray-600 dark:text-gray-300">{day.day}</div>
                    <div className="flex-1 bg-gray-200 dark:bg-dark-700 rounded-full h-2">
                      <div
                        className="bg-primary-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${(day.sessions / 6) * 100}%` }}
                      />
                    </div>
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-300 w-16 text-right">
                    {day.sessions} sessions
                  </div>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Emotions Breakdown */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="card"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Emotions Breakdown</h3>
            <div className="space-y-3">
              {Object.entries(analytics.emotionsBreakdown).map(([emotion, count]) => (
                <div key={emotion} className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className={`w-3 h-3 rounded-full ${getEmotionBgColor(emotion)}`} />
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300 capitalize">
                      {emotion}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-gray-200 dark:bg-dark-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${getEmotionColor(emotion).replace('text-', 'bg-')}`}
                        style={{ width: `${(count / Math.max(...Object.values(analytics.emotionsBreakdown))) * 100}%` }}
                      />
                    </div>
                    <span className="text-sm text-gray-600 dark:text-gray-300 w-8 text-right">
                      {count}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card"
        >
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Recent Activity</h3>
            <button className="btn-outline flex items-center space-x-2">
              <RefreshCw className="h-4 w-4" />
              <span>Refresh</span>
            </button>
          </div>
          <div className="space-y-4">
            {analytics.recentActivity.map((activity) => (
              <div key={activity.id} className="flex items-center space-x-4 p-4 bg-gray-50 dark:bg-dark-700 rounded-lg">
                <div className={`h-10 w-10 rounded-full flex items-center justify-center ${
                  activity.type === 'emotion' 
                    ? getEmotionBgColor(activity.emotion)
                    : 'bg-primary-100 dark:bg-primary-900/30'
                }`}>
                  {activity.type === 'emotion' ? (
                    <Brain className={`h-5 w-5 ${getEmotionColor(activity.emotion)}`} />
                  ) : (
                    <MessageCircle className="h-5 w-5 text-primary-600 dark:text-primary-400" />
                  )}
                </div>
                <div className="flex-1">
                  <p className="text-sm font-medium text-gray-900 dark:text-white">
                    {activity.type === 'emotion' 
                      ? `Detected ${activity.emotion} emotion (${activity.confidence}% confidence)`
                      : activity.message
                    }
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {activity.timestamp.toLocaleString()} • {formatTime(activity.duration)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Analytics;
