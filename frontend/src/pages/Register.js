import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import { useAuth } from '../contexts/AuthContext';
import { Eye, EyeOff, Lock, Mail, User, Brain, ArrowRight, CheckCircle, Shield, Users, TrendingUp, Sparkles } from 'lucide-react';
import ModernLoader from '../components/ui/ModernLoader';

const Register = () => {
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const { register: registerUser } = useAuth();
  const navigate = useNavigate();

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm();

  const password = watch('password');

  const onSubmit = async (data) => {
    setIsLoading(true);
    setError('');
    
    try {
      // Split the name into firstName and lastName
      const nameParts = data.name.trim().split(' ');
      const firstName = nameParts[0];
      const lastName = nameParts.slice(1).join(' ') || firstName; // Use firstName as lastName if only one name provided
      
      // Create userData object with the correct structure
      const emailPrefix = data.email.split('@')[0];
      // Clean username to only contain letters, numbers, and underscores
      const username = emailPrefix.replace(/[^a-zA-Z0-9_]/g, '').substring(0, 30);
      
      const userData = {
        firstName,
        lastName,
        username: username || `user_${Date.now()}`, // Fallback if email prefix is invalid
        email: data.email,
        password: data.password
      };
      
      const result = await registerUser(userData);
      if (result.success) {
        // Navigate to OTP verification page with state
        navigate('/verify-otp', { 
          state: { 
            userId: result.userId, 
            userEmail: data.email 
          } 
        });
      } else {
        setError(result.error || 'Registration failed');
      }
    } catch (err) {
      setError(err.message || 'Registration failed');
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-secondary-50/30 dark:from-dark-900 dark:via-dark-800 dark:to-dark-900 relative overflow-hidden">
      {/* Animated Background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-grid-pattern opacity-5"></div>
        
        {/* Floating Orbs */}
        <div className="absolute top-20 right-10 w-32 h-32 bg-gradient-to-r from-secondary-400/20 to-accent-400/20 rounded-full blur-xl animate-float"></div>
        <div className="absolute top-40 left-20 w-24 h-24 bg-gradient-to-r from-accent-400/20 to-primary-400/20 rounded-full blur-xl animate-float" style={{ animationDelay: '1s' }}></div>
        <div className="absolute bottom-20 right-1/4 w-40 h-40 bg-gradient-to-r from-primary-400/20 to-secondary-400/20 rounded-full blur-xl animate-float" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 min-h-screen flex items-center justify-center p-4">
        <motion.div
          className="w-full max-w-md"
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          {/* Header */}
          <motion.div 
            className="text-center mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            <motion.div 
              className="inline-flex items-center space-x-3 bg-white/80 dark:bg-dark-800/80 backdrop-blur-xl text-secondary-700 dark:text-secondary-300 px-6 py-3 rounded-full text-sm font-semibold mb-8 border border-white/20 dark:border-dark-700/50 shadow-xl"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.3 }}
            >
              <Brain className="h-5 w-5" />
              <span>Join Our Community</span>
              <div className="w-2 h-2 bg-secondary-500 rounded-full animate-pulse"></div>
            </motion.div>
            
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-white mb-4">
              Create Your
              <span className="gradient-text block">Account</span>
            </h1>
            <p className="text-lg text-gray-600 dark:text-gray-300">
              Start your emotion analysis journey today
            </p>
          </motion.div>

          {/* Registration Form */}
          <motion.div 
            className="card-gradient"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
          >
            <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
              {/* Name Field */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                  Full Name
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <User className="h-5 w-5 text-secondary-500" />
                  </div>
                  <input
                    {...register('name', {
                      required: 'Name is required',
                      minLength: {
                        value: 2,
                        message: 'Name must be at least 2 characters'
                      }
                    })}
                    type="text"
                    className="input-field pl-12"
                    placeholder="Enter your full name"
                  />
                </div>
                <AnimatePresence>
                  {errors.name && (
                    <motion.p
                      className="mt-2 text-sm text-red-500 flex items-center space-x-1"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -10 }}
                    >
                      <span>⚠️</span>
                      <span>{errors.name.message}</span>
                    </motion.p>
                  )}
                </AnimatePresence>
              </div>

              {/* Email Field */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                  Email Address
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <Mail className="h-5 w-5 text-secondary-500" />
                  </div>
                  <input
                    {...register('email', {
                      required: 'Email is required',
                      pattern: {
                        value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                        message: 'Invalid email address'
                      }
                    })}
                    type="email"
                    className="input-field pl-12"
                    placeholder="Enter your email"
                  />
                </div>
                <AnimatePresence>
                  {errors.email && (
                    <motion.p
                      className="mt-2 text-sm text-red-500 flex items-center space-x-1"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -10 }}
                    >
                      <span>⚠️</span>
                      <span>{errors.email.message}</span>
                    </motion.p>
                  )}
                </AnimatePresence>
              </div>

              {/* Password Field */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                  Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-secondary-500" />
                  </div>
                  <input
                    {...register('password', {
                      required: 'Password is required',
                      minLength: {
                        value: 6,
                        message: 'Password must be at least 6 characters'
                      },
                      pattern: {
                        value: /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/,
                        message: 'Password must contain uppercase, lowercase, and number'
                      }
                    })}
                    type={showPassword ? 'text' : 'password'}
                    className="input-field pl-12 pr-12"
                    placeholder="Create a password"
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-4 flex items-center"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeOff className="h-5 w-5 text-gray-400 hover:text-gray-600 transition-colors" />
                    ) : (
                      <Eye className="h-5 w-5 text-gray-400 hover:text-gray-600 transition-colors" />
                    )}
                  </button>
                </div>
                <AnimatePresence>
                  {errors.password && (
                    <motion.p
                      className="mt-2 text-sm text-red-500 flex items-center space-x-1"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -10 }}
                    >
                      <span>⚠️</span>
                      <span>{errors.password.message}</span>
                    </motion.p>
                  )}
                </AnimatePresence>
              </div>

              {/* Confirm Password Field */}
              <div>
                <label className="block text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                  Confirm Password
                </label>
                <div className="relative">
                  <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                    <Lock className="h-5 w-5 text-secondary-500" />
                  </div>
                  <input
                    {...register('confirmPassword', {
                      required: 'Please confirm your password',
                      validate: value => value === password || 'Passwords do not match'
                    })}
                    type={showConfirmPassword ? 'text' : 'password'}
                    className="input-field pl-12 pr-12"
                    placeholder="Confirm your password"
                  />
                  <button
                    type="button"
                    className="absolute inset-y-0 right-0 pr-4 flex items-center"
                    onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  >
                    {showConfirmPassword ? (
                      <EyeOff className="h-5 w-5 text-gray-400 hover:text-gray-600 transition-colors" />
                    ) : (
                      <Eye className="h-5 w-5 text-gray-400 hover:text-gray-600 transition-colors" />
                    )}
                  </button>
                </div>
                <AnimatePresence>
                  {errors.confirmPassword && (
                    <motion.p
                      className="mt-2 text-sm text-red-500 flex items-center space-x-1"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -10 }}
                    >
                      <span>⚠️</span>
                      <span>{errors.confirmPassword.message}</span>
                    </motion.p>
                  )}
                </AnimatePresence>
              </div>

              {/* Password Requirements */}
              <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 border border-blue-200 dark:border-blue-800 rounded-xl p-4">
                <p className="text-sm font-semibold text-blue-700 dark:text-blue-300 mb-3 flex items-center space-x-2">
                  <Shield className="w-4 h-4" />
                  <span>Password Requirements:</span>
                </p>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {[
                    { text: 'At least 6 characters', valid: password?.length >= 6 },
                    { text: 'One uppercase letter', valid: /[A-Z]/.test(password || '') },
                    { text: 'One lowercase letter', valid: /[a-z]/.test(password || '') },
                    { text: 'One number', valid: /\d/.test(password || '') }
                  ].map(({ text, valid }, i) => (
                    <motion.div
                      key={i}
                      className="flex items-center space-x-2"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                    >
                      <motion.div
                        animate={{ scale: valid ? 1.2 : 1 }}
                        transition={{ duration: 0.2 }}
                      >
                        <CheckCircle className={`w-3 h-3 ${valid ? 'text-green-500' : 'text-gray-400'}`} />
                      </motion.div>
                      <span className={valid ? 'text-green-600 dark:text-green-400' : 'text-gray-500 dark:text-gray-400'}>{text}</span>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Error Message */}
              <AnimatePresence>
                {error && (
                  <motion.div
                    className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl"
                    initial={{ opacity: 0, y: -10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                  >
                    <p className="text-sm text-red-600 dark:text-red-400 flex items-center space-x-2">
                      <span>🚨</span>
                      <span>{error}</span>
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Submit Button */}
              <motion.button
                type="submit"
                disabled={isLoading}
                className="btn-primary w-full flex items-center justify-center space-x-3"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                {isLoading ? (
                  <ModernLoader size="sm" />
                ) : (
                  <>
                    <span>Create Account</span>
                    <ArrowRight className="h-5 w-5" />
                  </>
                )}
              </motion.button>
            </form>

            {/* Divider */}
            <div className="mt-8 mb-8">
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-gray-200 dark:border-dark-700" />
                </div>
                <div className="relative flex justify-center text-sm">
                  <span className="px-4 bg-white dark:bg-dark-800 text-gray-500 dark:text-gray-400">
                    Already have an account?
                  </span>
                </div>
              </div>
            </div>

            {/* Login Link */}
            <div className="text-center">
              <Link
                to="/login"
                className="inline-flex items-center space-x-2 text-secondary-600 dark:text-secondary-400 hover:text-secondary-700 dark:hover:text-secondary-300 font-semibold transition-colors duration-200 group"
              >
                <span>Sign in instead</span>
                <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </motion.div>

          {/* Footer with stats */}
          <motion.div
            className="text-center mt-12"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6, delay: 0.8 }}
          >
            <div className="flex justify-center space-x-8 text-sm text-gray-500 dark:text-gray-400">
              <div className="flex items-center space-x-2">
                <Users className="h-4 w-4" />
                <span className="font-semibold">10K+ Users</span>
              </div>
              <div className="flex items-center space-x-2">
                <TrendingUp className="h-4 w-4" />
                <span className="font-semibold">99.9% Accuracy</span>
              </div>
              <div className="flex items-center space-x-2">
                <Shield className="h-4 w-4" />
                <span className="font-semibold">Secure</span>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default Register;