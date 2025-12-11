import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import OTPVerification from '../components/auth/OTPVerification';

const VerifyOTP = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const [userId, setUserId] = useState(null);
  const [userEmail, setUserEmail] = useState('');

  useEffect(() => {
    // Get userId and userEmail from navigation state
    if (location.state?.userId && location.state?.userEmail) {
      setUserId(location.state.userId);
      setUserEmail(location.state.userEmail);
    } else {
      // If no state, redirect to register with a message
      navigate('/register', { 
        state: { 
          message: 'Please complete registration first' 
        } 
      });
    }
  }, [location.state, navigate]);

  const handleBackToRegistration = () => {
    navigate('/register');
  };

  if (!userId || !userEmail) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary-500 border-t-transparent mx-auto mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <OTPVerification
      userId={userId}
      userEmail={userEmail}
      onBack={handleBackToRegistration}
    />
  );
};

export default VerifyOTP;
