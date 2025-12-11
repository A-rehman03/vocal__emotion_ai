const express = require('express');
const jwt = require('jsonwebtoken');
const { body, validationResult } = require('express-validator');
const User = require('../models/User');
const { protect } = require('../middleware/auth');
const crypto = require('crypto');
const { sendOTPEmail, sendPasswordResetEmail } = require('../services/emailService');

const router = express.Router();

// Generate JWT Token
const generateToken = (id) => {
  return jwt.sign({ id }, process.env.JWT_SECRET || 'fallback_secret_dev_only', {
    expiresIn: process.env.JWT_EXPIRE || '7d'
  });
};

// @route   POST /api/auth/register
// @desc    Register a new user (send OTP)
// @access  Public
router.post('/register', [
  body('username')
    .isLength({ min: 3, max: 30 })
    .withMessage('Username must be between 3 and 30 characters')
    .matches(/^[a-zA-Z0-9_]+$/)
    .withMessage('Username can only contain letters, numbers, and underscores'),
  body('email')
    .isEmail()
    .normalizeEmail()
    .withMessage('Please provide a valid email'),
  body('password')
    .isLength({ min: 6 })
    .withMessage('Password must be at least 6 characters long'),
  body('firstName')
    .isLength({ min: 1, max: 50 })
    .withMessage('First name is required and must be less than 50 characters'),
  body('lastName')
    .isLength({ min: 1, max: 50 })
    .withMessage('Last name is required and must be less than 50 characters')
], async (req, res) => {
  try {
    // Check for validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        message: 'Validation failed',
        errors: errors.array()
      });
    }

    const { username, email, password, firstName, lastName } = req.body;

    // Check if user already exists
    const existingUser = await User.findOne({
      $or: [{ email }, { username }]
    });

    if (existingUser) {
      if (existingUser.email === email) {
        return res.status(400).json({ message: 'Email already registered' });
      }
      if (existingUser.username === username) {
        return res.status(400).json({ message: 'Username already taken' });
      }
    }

    // Generate OTP
    const otp = crypto.randomInt(100000, 999999).toString();
    const otpExpiry = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

    // Create new user (not activated yet)
    const user = new User({
      username,
      email,
      password,
      firstName,
      lastName,
      isActive: false,
      otp,
      otpExpiry
    });

    await user.save();

    // Send OTP email
    const emailResult = await sendOTPEmail(email, otp, firstName);

    if (emailResult.success) {
      res.status(201).json({
        message: 'Registration successful. Please check your email for the verification code.',
        userId: user._id
      });
    } else {
      // If email fails, still return success but include OTP for development
      console.error('Email sending failed:', emailResult.error);
      res.status(201).json({
        message: 'Registration successful. Please verify your email with the OTP sent.',
        otp, // Include OTP in development if email fails
        userId: user._id
      });
    }

  } catch (error) {
    console.error('Registration error:', error);
    res.status(500).json({ message: 'Server error during registration' });
  }
});

// @route   POST /api/auth/verify-otp
// @desc    Verify OTP and activate user
// @access  Public
router.post('/verify-otp', [
  body('userId')
    .notEmpty()
    .withMessage('User ID is required'),
  body('otp')
    .isLength({ min: 6, max: 6 })
    .withMessage('OTP must be 6 digits')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        message: 'Validation failed',
        errors: errors.array()
      });
    }

    const { userId, otp } = req.body;

    const user = await User.findById(userId).select('+otp +otpExpiry');
    if (!user) {
      return res.status(400).json({ message: 'Invalid user ID' });
    }

    if (user.isActive) {
      return res.status(400).json({ message: 'User already verified' });
    }

    // Convert both to strings for comparison to avoid type issues
    if (String(user.otp) !== String(otp)) {
      return res.status(400).json({ message: 'Invalid OTP' });
    }

    if (new Date() > user.otpExpiry) {
      return res.status(400).json({ message: 'OTP expired' });
    }

    // Activate user
    user.isActive = true;
    user.otp = undefined;
    user.otpExpiry = undefined;
    await user.save();

    // Generate token
    const token = generateToken(user._id);

    res.json({
      message: 'Email verified successfully',
      token,
      user: user.getProfile()
    });

  } catch (error) {
    console.error('OTP verification error:', error);
    res.status(500).json({ message: 'Server error during verification' });
  }
});

// @route   POST /api/auth/resend-otp
// @desc    Resend OTP
// @access  Public
router.post('/resend-otp', [
  body('userId')
    .notEmpty()
    .withMessage('User ID is required')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        message: 'Validation failed',
        errors: errors.array()
      });
    }

    const { userId } = req.body;

    const user = await User.findById(userId).select('+otp +otpExpiry');
    if (!user) {
      return res.status(400).json({ message: 'Invalid user ID' });
    }

    if (user.isActive) {
      return res.status(400).json({ message: 'User already verified' });
    }

    // Generate new OTP
    const otp = crypto.randomInt(100000, 999999).toString();
    const otpExpiry = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

    user.otp = otp;
    user.otpExpiry = otpExpiry;
    await user.save();

    // Send OTP email
    const emailResult = await sendOTPEmail(user.email, otp, user.firstName);

    if (emailResult.success) {
      res.json({
        message: 'OTP resent successfully. Please check your email.'
      });
    } else {
      // If email fails, still return success but include OTP for development
      console.error('Email sending failed:', emailResult.error);
      res.json({
        message: 'OTP resent successfully',
        otp // Include OTP in development if email fails
      });
    }

  } catch (error) {
    console.error('Resend OTP error:', error);
    res.status(500).json({ message: 'Server error while resending OTP' });
  }
});

// @route   POST /api/auth/login
// @desc    Authenticate user & get token
// @access  Public
router.post('/login', [
  body('email')
    .isEmail()
    .normalizeEmail()
    .withMessage('Please provide a valid email'),
  body('password')
    .notEmpty()
    .withMessage('Password is required')
], async (req, res) => {
  try {
    // Check for validation errors
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        message: 'Validation failed',
        errors: errors.array()
      });
    }

    const { email, password } = req.body;

    // Check if user exists and password is correct
    const user = await User.findOne({ email }).select('+password');

    if (!user) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    if (!user.isActive) {
      return res.status(401).json({ message: 'Account is deactivated' });
    }

    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
      return res.status(401).json({ message: 'Invalid credentials' });
    }

    // Update last login
    user.lastLogin = new Date();
    await user.save();

    // Generate token
    const token = generateToken(user._id);

    res.json({
      message: 'Login successful',
      token,
      user: user.getProfile()
    });

  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ message: 'Server error during login' });
  }
});

// @route   GET /api/auth/me
// @desc    Get current user profile
// @access  Private
router.get('/me', protect, async (req, res) => {
  try {
    res.json({
      user: req.user.getProfile()
    });
  } catch (error) {
    console.error('Get profile error:', error);
    res.status(500).json({ message: 'Server error while fetching profile' });
  }
});

// @route   POST /api/auth/refresh
// @desc    Refresh JWT token
// @access  Private
router.post('/refresh', protect, async (req, res) => {
  try {
    const token = generateToken(req.user._id);
    res.json({
      message: 'Token refreshed successfully',
      token,
      user: req.user.getProfile()
    });
  } catch (error) {
    console.error('Token refresh error:', error);
    res.status(500).json({ message: 'Server error while refreshing token' });
  }
});

// @route   POST /api/auth/logout
// @desc    Logout user (client should remove token)
// @access  Private
router.post('/logout', protect, async (req, res) => {
  try {
    // In a more complex system, you might want to blacklist the token
    res.json({ message: 'Logged out successfully' });
  } catch (error) {
    console.error('Logout error:', error);
    res.status(500).json({ message: 'Server error during logout' });
  }
});

// @route   POST /api/auth/forgot-password
// @desc    Send password reset email
// @access  Public
router.post('/forgot-password', [
  body('email')
    .isEmail()
    .normalizeEmail()
    .withMessage('Please provide a valid email')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({
        message: 'Validation failed',
        errors: errors.array()
      });
    }

    const { email } = req.body;
    const user = await User.findOne({ email });

    if (!user) {
      // Don't reveal if email exists or not for security
      return res.json({ message: 'If an account with that email exists, a password reset link has been sent' });
    }

    // Generate reset token
    const resetToken = jwt.sign(
      { id: user._id, type: 'password-reset' },
      process.env.JWT_SECRET,
      { expiresIn: '1h' }
    );

    // Send password reset email
    const emailResult = await sendPasswordResetEmail(email, resetToken, user.firstName);

    if (emailResult.success) {
      res.json({
        message: 'If an account with that email exists, a password reset link has been sent'
      });
    } else {
      // If email fails, still return success but include token for development
      console.error('Email sending failed:', emailResult.error);
      res.json({
        message: 'If an account with that email exists, a password reset link has been sent',
        resetToken // Include token in development if email fails
      });
    }

  } catch (error) {
    console.error('Forgot password error:', error);
    res.status(500).json({ message: 'Server error while processing request' });
  }
});

module.exports = router;
