import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { useTheme } from '../../contexts/ThemeContext';
import { Mail, ArrowLeft, RefreshCw, CheckCircle, AlertCircle, Shield, Sparkles, Zap } from 'lucide-react';
import ModernLoader from '../ui/ModernLoader';

const OTPVerification = ({ userId, userEmail, onBack }) => {
  const [otp, setOtp] = useState(['', '', '', '', '', '']);
  const [isLoading, setIsLoading] = useState(false);
  const [isResending, setIsResending] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState(false);
  const [timeLeft, setTimeLeft] = useState(600); // 10 minutes in seconds
  const [isHovered, setIsHovered] = useState(false);
  const { verifyOTP, resendOTP } = useAuth();
  const { isDarkMode } = useTheme();
  const navigate = useNavigate();
  const inputRefs = useRef([]);

  // Timer countdown
  useEffect(() => {
    if (timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(timeLeft - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [timeLeft]);

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Floating animations
  const floatingVariants = {
    animate: (i) => ({
      y: [-15, 15, -15],
      x: [-8, 8, -8],
      rotate: [0, 180, 360],
      scale: [1, 1.1, 1],
      transition: {
        duration: 6 + i * 2,
        repeat: Infinity,
        ease: "easeInOut",
        delay: i * 0.5
      }
    })
  };

  // Container animations
  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
        delayChildren: 0.3
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 30, scale: 0.9 },
    visible: {
      opacity: 1,
      y: 0,
      scale: 1,
      transition: {
        duration: 0.6,
        ease: [0.25, 0.46, 0.45, 0.94]
      }
    }
  };

  const handleOtpChange = (index, value) => {
    // Only allow single digit and numbers only
    if (value.length > 1 || !/^\d*$/.test(value)) return;
    
    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);
    setError('');

    // Auto-focus next input
    if (value && index < 5) {
      inputRefs.current[index + 1]?.focus();
    }
  };

  const handleKeyDown = (index, e) => {
    if (e.key === 'Backspace' && !otp[index] && index > 0) {
      inputRefs.current[index - 1]?.focus();
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const otpString = otp.join('');
    
    if (otpString.length !== 6) {
      setError('Please enter all 6 digits');
      return;
    }

    setIsLoading(true);
    setError('');

    try {
      const result = await verifyOTP(userId, otpString);
      if (result.success) {
        setSuccess(true);
        setTimeout(() => {
          navigate('/dashboard');
        }, 2000);
      } else {
        setError(result.error || 'Invalid OTP. Please try again.');
        setOtp(['', '', '', '', '', '']);
        inputRefs.current[0]?.focus();
      }
    } catch (err) {
      setError(err.message || 'Invalid OTP. Please try again.');
      setOtp(['', '', '', '', '', '']);
      inputRefs.current[0]?.focus();
    } finally {
      setIsLoading(false);
    }
  };

  const handleResend = async () => {
    setIsResending(true);
    setError('');

    try {
      await resendOTP(userId);
      setTimeLeft(600); // Reset timer
      setOtp(['', '', '', '', '', '']);
      inputRefs.current[0]?.focus();
    } catch (err) {
      setError(err.message || 'Failed to resend OTP. Please try again.');
    } finally {
      setIsResending(false);
    }
  };

  return (
    <div 
      className="min-h-screen bg-gradient-to-br from-slate-900 via-indigo-900 to-slate-900 dark:from-dark-900 dark:via-indigo-900 dark:to-dark-900 relative overflow-hidden"
      style={{
        background: isDarkMode 
          ? 'radial-gradient(ellipse at top, #312e81 0%, #0f0f23 50%, #000000 100%)'
          : 'radial-gradient(ellipse at top, #4338ca 0%, #312e81 50%, #0f0f23 100%)'
      }}
    >
      {/* Animated Background */}
      <div className="absolute inset-0">
        {/* Mesh gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-r from-indigo-600/20 via-purple-600/20 to-pink-600/20" />
        
        {/* Floating geometric shapes */}
        <motion.div
          className="absolute top-20 left-20 w-32 h-32 bg-gradient-to-r from-indigo-500/30 to-purple-500/30 rounded-full blur-xl"
          animate={{
            x: [0, 100, 0],
            y: [0, -50, 0],
            scale: [1, 1.2, 1],
            rotate: [0, 180, 360]
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />
        <motion.div
          className="absolute top-40 right-32 w-24 h-24 bg-gradient-to-r from-purple-500/30 to-pink-500/30 rounded-full blur-lg"
          animate={{
            x: [0, -80, 0],
            y: [0, 60, 0],
            scale: [1, 0.8, 1],
            rotate: [0, -180, -360]
          }}
          transition={{
            duration: 15,
            repeat: Infinity,
            ease: "easeInOut"
          }}
        />

        {/* Floating Icons */}
        {[
          { Icon: Shield, delay: 0, color: 'text-indigo-400/60' },
          { Icon: CheckCircle, delay: 1, color: 'text-green-400/60' },
          { Icon: Sparkles, delay: 2, color: 'text-yellow-400/60' },
          { Icon: Zap, delay: 3, color: 'text-purple-400/60' }
        ].map(({ Icon, delay, color }, i) => (
          <motion.div
            key={i}
            className={`absolute ${color}`}
            style={{
              top: `${20 + i * 20}%`,
              left: `${10 + i * 15}%`,
            }}
            custom={i}
            variants={floatingVariants}
            animate="animate"
          >
            <Icon size={24 + i * 2} />
          </motion.div>
        ))}
      </div>

      {/* Main Content */}
      <div className="relative z-10 min-h-screen flex items-center justify-center p-4">
        <motion.div
          className="w-full max-w-md"
          variants={containerVariants}
          initial="hidden"
          animate="visible"
        >
          {/* Header */}
          <motion.div className="text-center mb-12" variants={itemVariants}>
            <motion.div
              className="relative inline-block"
              whileHover={{ scale: 1.05 }}
              onHoverStart={() => setIsHovered(true)}
              onHoverEnd={() => setIsHovered(false)}
            >
              <motion.div
                className="inline-flex items-center justify-center w-24 h-24 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full shadow-2xl"
                animate={{
                  rotate: isHovered ? 360 : 0,
                  scale: isHovered ? 1.1 : 1
                }}
                transition={{ duration: 0.6, ease: "easeInOut" }}
              >
                <Mail className="w-12 h-12 text-white" />
              </motion.div>
              
              {/* Glow effect */}
              <motion.div
                className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-full blur-xl opacity-50"
                animate={{
                  scale: isHovered ? 1.5 : 1.2,
                  opacity: isHovered ? 0.8 : 0.5
                }}
                transition={{ duration: 0.3 }}
              />
            </motion.div>

            <motion.h1
              className="text-4xl font-bold bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mt-8 mb-4"
              initial={{ opacity: 0, y: -30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6, duration: 0.8 }}
            >
              Verify Your Email
            </motion.h1>
            <motion.p
              className="text-gray-300 text-lg"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8, duration: 0.8 }}
            >
              We've sent a 6-digit code to
            </motion.p>
            <motion.p
              className="text-indigo-300 font-semibold"
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1, duration: 0.8 }}
            >
              {userEmail}
            </motion.p>
          </motion.div>

          {/* OTP Form */}
          <motion.div
            className="relative"
            variants={itemVariants}
            whileHover={{ y: -5 }}
            transition={{ duration: 0.3 }}
          >
            {/* Glassmorphism background */}
            <div className="absolute inset-0 bg-white/10 dark:bg-white/5 backdrop-blur-xl rounded-3xl border border-white/20 dark:border-white/10 shadow-2xl" />
            
            {/* Animated border */}
            <motion.div
              className="absolute inset-0 rounded-3xl"
              style={{
                background: 'linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6)',
                padding: '2px'
              }}
              animate={{
                background: [
                  'linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6)',
                  'linear-gradient(135deg, #8b5cf6, #ec4899, #3b82f6, #8b5cf6)',
                  'linear-gradient(225deg, #ec4899, #3b82f6, #8b5cf6, #ec4899)',
                  'linear-gradient(315deg, #3b82f6, #8b5cf6, #ec4899, #3b82f6)'
                ]
              }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
            >
              <div className="w-full h-full bg-slate-900/90 dark:bg-slate-900/95 rounded-3xl" />
            </motion.div>

            <div className="relative p-8">
              <form onSubmit={handleSubmit} className="space-y-8">
                {/* OTP Input */}
                <motion.div variants={itemVariants}>
                  <label className="block text-sm font-semibold text-gray-200 mb-4 text-center">
                    Enter Verification Code
                  </label>
                  <div className="flex justify-center space-x-3">
                    {otp.map((digit, index) => (
                      <motion.input
                        key={index}
                        ref={(el) => (inputRefs.current[index] = el)}
                        type="tel"
                        inputMode="numeric"
                        pattern="[0-9]*"
                        maxLength="1"
                        value={digit}
                        onChange={(e) => handleOtpChange(index, e.target.value)}
                        onKeyDown={(e) => handleKeyDown(index, e)}
                        className="w-12 h-12 text-center text-2xl font-bold bg-white/10 dark:bg-white/5 border border-white/20 dark:border-white/10 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-white placeholder-gray-400 transition-all duration-300 backdrop-blur-sm"
                        placeholder="0"
                        disabled={isLoading}
                        whileFocus={{ scale: 1.1 }}
                        whileHover={{ scale: 1.05 }}
                      />
                    ))}
                  </div>
                </motion.div>

                {/* Timer */}
                <motion.div
                  className="text-center"
                  variants={itemVariants}
                >
                  <div className="flex items-center justify-center space-x-2 text-sm text-gray-400">
                    <span>Code expires in:</span>
                    <motion.span
                      className={`font-mono font-bold ${timeLeft < 60 ? 'text-red-400' : 'text-indigo-400'}`}
                      animate={{ scale: timeLeft < 60 ? [1, 1.1, 1] : 1 }}
                      transition={{ duration: 0.5, repeat: timeLeft < 60 ? Infinity : 0 }}
                    >
                      {formatTime(timeLeft)}
                    </motion.span>
                  </div>
                </motion.div>

                {/* Error Message */}
                <AnimatePresence>
                  {error && (
                    <motion.div
                      className="p-4 bg-red-500/20 border border-red-500/30 rounded-2xl backdrop-blur-sm"
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    >
                      <p className="text-sm text-red-300 flex items-center space-x-2">
                        <AlertCircle className="w-4 h-4" />
                        <span>{error}</span>
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Success Message */}
                <AnimatePresence>
                  {success && (
                    <motion.div
                      className="p-4 bg-green-500/20 border border-green-500/30 rounded-2xl backdrop-blur-sm"
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    >
                      <p className="text-sm text-green-300 flex items-center space-x-2">
                        <CheckCircle className="w-4 h-4" />
                        <span>Email verified successfully! Redirecting...</span>
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Submit Button */}
                <motion.button
                  type="submit"
                  disabled={isLoading || otp.join('').length !== 6}
                  className="relative w-full group overflow-hidden"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  variants={itemVariants}
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 rounded-2xl blur-sm opacity-75 group-hover:opacity-100 transition-opacity duration-300" />
                  <div className="relative w-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:from-indigo-600 hover:via-purple-600 hover:to-pink-600 text-white font-bold py-4 px-6 rounded-2xl transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-3 shadow-2xl">
                    {isLoading ? (
                      <ModernLoader size="sm" />
                    ) : (
                      <>
                        <span>Verify Email</span>
                        <motion.div
                          animate={{ x: [0, 5, 0] }}
                          transition={{ duration: 1, repeat: Infinity }}
                        >
                          <CheckCircle className="w-5 h-5" />
                        </motion.div>
                      </>
                    )}
                  </div>
                </motion.button>
              </form>

              {/* Resend Section */}
              <motion.div
                className="mt-8 text-center"
                variants={itemVariants}
              >
                <p className="text-sm text-gray-400 mb-4">
                  Didn't receive the code?
                </p>
                <motion.button
                  onClick={handleResend}
                  disabled={isResending || timeLeft > 0}
                  className="text-indigo-400 hover:text-indigo-300 font-semibold transition-colors duration-200 flex items-center justify-center space-x-2 mx-auto disabled:opacity-50 disabled:cursor-not-allowed"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {isResending ? (
                    <ModernLoader size="sm" />
                  ) : (
                    <>
                      <RefreshCw className="w-4 h-4" />
                      <span>Resend Code</span>
                    </>
                  )}
                </motion.button>
              </motion.div>

              {/* Back Button */}
              <motion.div
                className="mt-6 text-center"
                variants={itemVariants}
              >
                <motion.button
                  onClick={onBack}
                  className="text-gray-400 hover:text-gray-300 font-medium transition-colors duration-200 flex items-center justify-center space-x-2 mx-auto"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <ArrowLeft className="w-4 h-4" />
                  <span>Back to Registration</span>
                </motion.button>
              </motion.div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default OTPVerification;