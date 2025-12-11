import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  Brain,
  Heart,
  Mic,
  MessageCircle,
  Zap,
  Shield,
  Users,
  ArrowRight,
  Play,
  Star,
  Sparkles,
  Globe,
  Award,
  CheckCircle
} from 'lucide-react';

import { features, stats, testimonials, howItWorksSteps } from '../data/landingPageData';

const Home = () => {
  // Data is now imported from ../data/landingPageData.js

  return (
    <div className="min-h-screen bg-transparent">
      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0 bg-gradient-to-br from-primary-50 via-white to-secondary-50 dark:from-dark-900 dark:via-dark-800 dark:to-dark-900"></div>
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>

        {/* Floating Orbs */}
        <div className="absolute top-20 left-10 w-32 h-32 bg-gradient-to-r from-primary-400/20 to-secondary-400/20 rounded-full blur-xl animate-float"></div>
        <div className="absolute top-40 right-20 w-24 h-24 bg-gradient-to-r from-secondary-400/20 to-accent-400/20 rounded-full blur-xl animate-float" style={{ animationDelay: '1s' }}></div>
        <div className="absolute bottom-20 left-1/4 w-40 h-40 bg-gradient-to-r from-accent-400/20 to-primary-400/20 rounded-full blur-xl animate-float" style={{ animationDelay: '2s' }}></div>

        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.div
            className="text-center max-w-5xl mx-auto"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Badge */}
            <motion.div
              className="inline-flex items-center space-x-2 bg-white/80 dark:bg-dark-800/80 backdrop-blur-xl text-primary-700 dark:text-primary-300 px-6 py-3 rounded-full text-sm font-semibold mb-8 border border-white/20 dark:border-dark-700/50 shadow-xl"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <Sparkles className="h-4 w-4" />
              <span>AI-Powered Emotion Recognition</span>
              <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse"></div>
            </motion.div>

            {/* Main Heading */}
            <motion.h1
              className="text-5xl md:text-7xl font-bold text-gray-900 dark:text-white mb-8 leading-tight"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
            >
              Understand Emotions Through
              <span className="gradient-text block mt-2">Voice & AI</span>
            </motion.h1>

            {/* Subtitle */}
            <motion.p
              className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              Experience the future of human-computer interaction with our advanced speech emotion recognition technology.
              Get emotionally intelligent responses that truly understand how you feel.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-16"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.5 }}
            >
              <Link
                to="/register"
                className="btn-primary px-10 py-4 text-lg font-bold flex items-center space-x-3 group hover-lift hover-glow"
              >
                <span>Get Started Free</span>
                <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform duration-200" />
              </Link>
              <button className="btn-outline px-10 py-4 text-lg font-bold flex items-center space-x-3 hover-lift">
                <Play className="h-5 w-5" />
                <span>Watch Demo</span>
              </button>
            </motion.div>

            {/* Trust Indicators */}
            <motion.div
              className="flex flex-wrap justify-center items-center gap-8 text-sm text-gray-500 dark:text-gray-400"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.8, delay: 0.6 }}
            >
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>99.2% Accuracy</span>
              </div>
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>Real-time Processing</span>
              </div>
              <div className="flex items-center space-x-2">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <span>Privacy First</span>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-20 bg-gradient-to-br from-white via-gray-50 to-white dark:from-dark-900 dark:via-dark-800 dark:to-dark-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="grid grid-cols-2 md:grid-cols-4 gap-8"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            {stats.map((stat, index) => (
              <motion.div
                key={index}
                className="text-center card-glass hover-lift"
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.4, delay: index * 0.1 }}
              >
                <div className="text-4xl md:text-5xl font-bold gradient-text mb-3">
                  {stat.number}
                </div>
                <div className="text-gray-600 dark:text-gray-300 font-semibold text-lg">
                  {stat.label}
                </div>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-gradient-to-br from-gray-50 via-white to-gray-50 dark:from-dark-800 dark:via-dark-900 dark:to-dark-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-20"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
              Powerful Features for
              <span className="gradient-text block">Emotion Recognition</span>
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Our cutting-edge technology combines advanced AI with sophisticated speech analysis
              to deliver unparalleled emotion recognition capabilities.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={index}
                  className="card-gradient group hover-lift"
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.1 }}
                >
                  <div className="flex items-center justify-between mb-6">
                    <div className={`h-16 w-16 bg-gradient-to-br from-${feature.color}-500 to-${feature.color}-600 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform duration-300 shadow-lg`}>
                      <Icon className="h-8 w-8 text-white" />
                    </div>
                    <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full opacity-20 group-hover:opacity-40 transition-opacity duration-300"></div>
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                    {feature.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 leading-relaxed text-lg">
                    {feature.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-24 bg-gradient-to-br from-white via-primary-50/30 to-secondary-50/30 dark:from-dark-900 dark:via-dark-800 dark:to-dark-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-20"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
              How It Works
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
              Experience emotion recognition in three simple steps
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-12">
            {howItWorksSteps.map((item, index) => {
              const Icon = item.icon;
              return (
                <motion.div
                  key={index}
                  className="text-center card-gradient hover-lift"
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ duration: 0.5, delay: index * 0.2 }}
                >
                  <div className="relative mb-8">
                    <div className={`h-20 w-20 bg-gradient-to-br from-${item.color}-500 to-${item.color}-600 rounded-3xl flex items-center justify-center mx-auto shadow-xl`}>
                      <Icon className="h-10 w-10 text-white" />
                    </div>
                    <div className={`absolute -top-2 -right-2 h-8 w-8 bg-gradient-to-r from-${item.color}-400 to-${item.color}-500 rounded-full flex items-center justify-center text-white font-bold text-lg shadow-lg`}>
                      {item.step}
                    </div>
                  </div>
                  <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
                    {item.title}
                  </h3>
                  <p className="text-gray-600 dark:text-gray-300 leading-relaxed text-lg">
                    {item.description}
                  </p>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-24 bg-gradient-to-br from-gray-50 via-white to-gray-50 dark:from-dark-800 dark:via-dark-900 dark:to-dark-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            className="text-center mb-20"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
          >
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-6">
              Loved by Users
              <span className="gradient-text block">Worldwide</span>
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto leading-relaxed">
              See what our users are saying about Vocal Emotion AI
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={index}
                className="card-gradient text-center hover-lift"
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
              >
                <div className="flex justify-center mb-6">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="h-6 w-6 text-yellow-400 fill-current" />
                  ))}
                </div>
                <p className="text-gray-600 dark:text-gray-300 mb-8 italic text-lg leading-relaxed">
                  "{testimonial.content}"
                </p>
                <div className="border-t border-gray-200 dark:border-dark-700 pt-6">
                  <div className="font-bold text-gray-900 dark:text-white text-lg">{testimonial.name}</div>
                  <div className="text-sm text-gray-500 dark:text-gray-400 font-medium">{testimonial.role}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-br from-primary-600 via-primary-700 to-secondary-600 relative overflow-hidden">
        {/* Background Effects */}
        <div className="absolute inset-0 bg-grid-pattern opacity-10"></div>
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-primary-600/90 to-secondary-600/90"></div>

        {/* Floating Elements */}
        <div className="absolute top-10 left-10 w-20 h-20 bg-white/10 rounded-full blur-xl animate-float"></div>
        <div className="absolute bottom-10 right-10 w-32 h-32 bg-white/5 rounded-full blur-2xl animate-float" style={{ animationDelay: '2s' }}></div>

        <div className="max-w-5xl mx-auto text-center px-4 sm:px-6 lg:px-8 relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8 }}
          >
            <h2 className="text-4xl md:text-6xl font-bold text-white mb-8 leading-tight">
              Ready to Experience
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-white to-yellow-200">
                Emotion Recognition?
              </span>
            </h2>
            <p className="text-xl md:text-2xl text-primary-100 mb-12 max-w-3xl mx-auto leading-relaxed">
              Join thousands of users who are already experiencing the future of AI-powered emotion recognition.
            </p>
            <div className="flex flex-col sm:flex-row gap-6 justify-center items-center">
              <Link
                to="/register"
                className="bg-white text-primary-600 hover:bg-gray-100 font-bold px-10 py-4 rounded-xl transition-all duration-300 flex items-center justify-center space-x-3 hover-lift shadow-xl"
              >
                <span>Start Free Trial</span>
                <ArrowRight className="h-5 w-5" />
              </Link>
              <Link
                to="/contact"
                className="border-2 border-white text-white hover:bg-white hover:text-primary-600 font-bold px-10 py-4 rounded-xl transition-all duration-300 hover-lift"
              >
                Contact Sales
              </Link>
            </div>

            {/* Trust Indicators */}
            <div className="mt-16 flex flex-wrap justify-center items-center gap-8 text-primary-100">
              <div className="flex items-center space-x-2">
                <Users className="h-5 w-5" />
                <span className="font-semibold">10,000+ Users</span>
              </div>
              <div className="flex items-center space-x-2">
                <Award className="h-5 w-5" />
                <span className="font-semibold">99.2% Accuracy</span>
              </div>
              <div className="flex items-center space-x-2">
                <Globe className="h-5 w-5" />
                <span className="font-semibold">Global Reach</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

export default Home;
