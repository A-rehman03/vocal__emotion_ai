import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../contexts/AuthContext';
import {
  Brain,
  MessageCircle,
  Mic,
  BarChart3,
  TrendingUp,
  Activity,
  Calendar,
  Clock,
  ArrowRight
} from 'lucide-react';

const Dashboard = () => {
  const { user } = useAuth();

  const quickActions = [
    {
      title: 'Start Chat',
      description: 'Begin a conversation with our AI chatbot',
      icon: MessageCircle,
      path: '/chat',
      color: 'primary',
      gradient: 'from-primary-500 to-primary-600'
    },
    {
      title: 'Emotion Analysis',
      description: 'Analyze voice emotions in real-time',
      icon: Mic,
      path: '/emotion-analysis',
      color: 'secondary',
      gradient: 'from-secondary-500 to-secondary-600'
    },
    {
      title: 'View Analytics',
      description: 'Check your emotion recognition history',
      icon: BarChart3,
      path: '/analytics',
      color: 'emotion-excited',
      gradient: 'from-emotion-excited to-orange-600'
    }
  ];

  const stats = [
    {
      label: 'Total Conversations',
      value: user?.stats?.conversations || '0',
      change: '+12%',
      changeType: 'positive',
      icon: MessageCircle
    },
    {
      label: 'Emotions Analyzed',
      value: user?.stats?.emotions || '0',
      change: '+8%',
      changeType: 'positive',
      icon: Brain
    },
    {
      label: 'Accuracy Rate',
      value: user?.stats?.accuracy || '0%',
      change: '+2.1%',
      changeType: 'positive',
      icon: TrendingUp
    },
    {
      label: 'Active Sessions',
      value: user?.stats?.activeSessions || '0',
      change: '0%',
      changeType: 'neutral',
      icon: Activity
    }
  ];

  const recentActivity = user?.recentActivity || [
    {
      type: 'welcome',
      description: 'Welcome to Vocal Emotion AI! Start by analyzing your voice or chatting with our AI.',
      time: 'Just now',
      icon: Brain
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-primary-50/30 dark:from-dark-900 dark:via-dark-800 dark:to-dark-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Welcome Header */}
        <motion.div
          className="mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="flex items-center justify-between mb-8">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
                Welcome back, {user?.firstName || user?.username}! 👋
              </h1>
              <p className="text-xl text-gray-600 dark:text-gray-300">
                Here's what's happening with your Vocal Emotion AI experience today.
              </p>
            </div>
            <div className="hidden lg:block">
              <div className="w-20 h-20 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-3xl flex items-center justify-center shadow-xl">
                <Brain className="h-10 w-10 text-white" />
              </div>
            </div>
          </div>
        </motion.div>

        {/* Stats Grid */}
        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={index}
                className="card-gradient hover-lift"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.4, delay: 0.1 * index }}
              >
                <div className="flex items-center justify-between mb-6">
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-gray-600 dark:text-gray-300 mb-2 uppercase tracking-wide">
                      {stat.label}
                    </p>
                    <p className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
                      {stat.value}
                    </p>
                    <div className="flex items-center">
                      <span className={`text-sm font-semibold ${stat.changeType === 'positive' ? 'text-green-600 dark:text-green-400' :
                          stat.changeType === 'negative' ? 'text-red-600 dark:text-red-400' : 'text-gray-600 dark:text-gray-400'
                        }`}>
                        {stat.change}
                      </span>
                      <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">from last week</span>
                    </div>
                  </div>
                  <div className={`h-16 w-16 bg-gradient-to-br from-${stat.color}-500 to-${stat.color}-600 rounded-2xl flex items-center justify-center shadow-lg`}>
                    <Icon className="h-8 w-8 text-white" />
                  </div>
                </div>
              </motion.div>
            );
          })}
        </motion.div>

        {/* Quick Actions */}
        <motion.div
          className="mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-8">
            Quick Actions
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {quickActions.map((action, index) => {
              const Icon = action.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.4, delay: 0.1 * index }}
                >
                  <Link
                    to={action.path}
                    className="card-gradient group cursor-pointer hover-lift block"
                  >
                    <div className="flex items-center justify-between mb-6">
                      <div className={`h-16 w-16 bg-gradient-to-r ${action.gradient} rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300 shadow-lg`}>
                        <Icon className="h-8 w-8 text-white" />
                      </div>
                      <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full opacity-20 group-hover:opacity-40 transition-opacity duration-300"></div>
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-3">
                      {action.title}
                    </h3>
                    <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                      {action.description}
                    </p>
                    <div className="mt-4 flex items-center text-primary-600 dark:text-primary-400 font-semibold group-hover:translate-x-2 transition-transform duration-300">
                      <span>Get Started</span>
                      <ArrowRight className="h-4 w-4 ml-2" />
                    </div>
                  </Link>
                </motion.div>
              );
            })}
          </div>
        </motion.div>

        {/* Recent Activity & Quick Stats */}
        <motion.div
          className="grid grid-cols-1 lg:grid-cols-2 gap-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          {/* Recent Activity */}
          <div className="card-gradient">
            <div className="flex items-center justify-between mb-8">
              <h3 className="text-xl font-bold text-gray-900 dark:text-white">
                Recent Activity
              </h3>
              <Link
                to="/analytics"
                className="text-sm text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 font-semibold flex items-center space-x-1 hover:translate-x-1 transition-transform duration-300"
              >
                <span>View all</span>
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
            <div className="space-y-6">
              {recentActivity.map((activity, index) => {
                const Icon = activity.icon;
                return (
                  <motion.div
                    key={index}
                    className="flex items-start space-x-4 p-4 bg-white/50 dark:bg-dark-800/50 rounded-xl hover:bg-white/80 dark:hover:bg-dark-800/80 transition-colors duration-300"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.4, delay: 0.1 * index }}
                  >
                    <div className="h-10 w-10 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-xl flex items-center justify-center flex-shrink-0 shadow-lg">
                      <Icon className="h-5 w-5 text-white" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 dark:text-white leading-relaxed">
                        {activity.description}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 font-semibold">
                        {activity.time}
                      </p>
                    </div>
                  </motion.div>

                );
              })}
            </div>
          </div>

          {/* Quick Stats */}
          <div className="card-gradient">
            <h3 className="text-xl font-bold text-gray-900 dark:text-white mb-8">
              This Week's Summary
            </h3>
            <div className="space-y-6">
              {[
                { icon: Calendar, label: "Days Active", value: "5/7", color: "primary" },
                { icon: Clock, label: "Total Time", value: "2h 34m", color: "secondary" },
                { icon: Brain, label: "AI Interactions", value: "47", color: "accent" },
                { icon: TrendingUp, label: "Improvement", value: "+15%", color: "green" }
              ].map((stat, index) => {
                const Icon = stat.icon;
                return (
                  <motion.div
                    key={index}
                    className="flex items-center justify-between p-4 bg-white/50 dark:bg-dark-800/50 rounded-xl hover:bg-white/80 dark:hover:bg-dark-800/80 transition-colors duration-300"
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.4, delay: 0.1 * index }}
                  >
                    <div className="flex items-center space-x-4">
                      <div className={`h-10 w-10 bg-gradient-to-br from-${stat.color}-500 to-${stat.color}-600 rounded-xl flex items-center justify-center shadow-lg`}>
                        <Icon className="h-5 w-5 text-white" />
                      </div>
                      <span className="text-sm font-semibold text-gray-600 dark:text-gray-300">{stat.label}</span>
                    </div>
                    <span className={`text-lg font-bold ${stat.color === 'green' ? 'text-green-600 dark:text-green-400' : 'text-gray-900 dark:text-white'
                      }`}>
                      {stat.value}
                    </span>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Getting Started Tips */}
      <motion.div
        className="mt-12 card-gradient border-2 border-primary-200/50 dark:border-primary-800/50"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, delay: 0.8 }}
      >
        <div className="flex items-start space-x-6">
          <div className="h-16 w-16 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-2xl flex items-center justify-center flex-shrink-0 shadow-lg">
            <Brain className="h-8 w-8 text-white" />
          </div>
          <div className="flex-1">
            <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
              Tips to Get Started
            </h3>
            <p className="text-gray-600 dark:text-gray-300 mb-6 text-lg leading-relaxed">
              New to Vocal Emotion AI? Here are some tips to get the most out of your experience:
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                "Try speaking naturally - our AI works best with conversational speech",
                "Use the emotion analysis feature to understand your vocal patterns",
                "Check your analytics to track your emotional journey over time",
                "Don't hesitate to provide feedback to help improve our AI"
              ].map((tip, index) => (
                <motion.div
                  key={index}
                  className="flex items-start space-x-3 p-3 bg-white/50 dark:bg-dark-800/50 rounded-xl"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.3, delay: 0.1 * index }}
                >
                  <div className="w-2 h-2 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full mt-2 flex-shrink-0"></div>
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300 leading-relaxed">
                    {tip}
                  </span>
                </motion.div>
              ))}
            </div>
          </div>
        </div>
      </motion.div>
    </div>

  );
};

export default Dashboard;