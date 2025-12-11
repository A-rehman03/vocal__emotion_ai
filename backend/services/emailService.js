const nodemailer = require('nodemailer');

// Create transporter for email sending
const createTransporter = () => {
  // For development, we'll use Gmail SMTP
  // In production, you should use a proper email service like SendGrid, AWS SES, etc.
  return nodemailer.createTransport({
    service: 'gmail',
    auth: {
      user: process.env.EMAIL_USER, // Your Gmail address
      pass: process.env.EMAIL_PASS  // Your Gmail app password
    }
  });
};

// Send OTP email
const sendOTPEmail = async (email, otp, firstName) => {
  try {
    const transporter = createTransporter();
    
    const mailOptions = {
      from: process.env.EMAIL_USER,
      to: email,
      subject: 'Verify Your Email - Vocal Emotion AI',
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
          <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #3b82f6; margin: 0;">Vocal Emotion AI</h1>
            <p style="color: #6b7280; margin: 5px 0 0 0;">AI-Powered Speech Emotion Recognition</p>
          </div>
          
          <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
            <h2 style="color: white; margin: 0 0 15px 0; font-size: 24px;">Welcome ${firstName}!</h2>
            <p style="color: white; margin: 0; font-size: 16px;">Please verify your email address to complete your registration</p>
          </div>
          
          <div style="background: #f8fafc; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 30px;">
            <p style="color: #374151; margin: 0 0 15px 0; font-size: 16px;">Your verification code is:</p>
            <div style="background: white; padding: 20px; border-radius: 8px; border: 2px dashed #3b82f6; display: inline-block;">
              <span style="font-size: 32px; font-weight: bold; color: #3b82f6; letter-spacing: 5px;">${otp}</span>
            </div>
            <p style="color: #6b7280; margin: 15px 0 0 0; font-size: 14px;">This code will expire in 10 minutes</p>
          </div>
          
          <div style="background: #fef3c7; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <p style="color: #92400e; margin: 0; font-size: 14px;">
              <strong>Security Note:</strong> Never share this code with anyone. Vocal Emotion AI will never ask for your verification code.
            </p>
          </div>
          
          <div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #e5e7eb;">
            <p style="color: #6b7280; margin: 0; font-size: 12px;">
              If you didn't create an account with Vocal Emotion AI, please ignore this email.
            </p>
          </div>
        </div>
      `
    };

    const result = await transporter.sendMail(mailOptions);
    console.log('OTP email sent successfully:', result.messageId);
    return { success: true, messageId: result.messageId };
  } catch (error) {
    console.error('Error sending OTP email:', error);
    return { success: false, error: error.message };
  }
};

// Send password reset email
const sendPasswordResetEmail = async (email, resetToken, firstName) => {
  try {
    const transporter = createTransporter();
    
    const resetUrl = `${process.env.FRONTEND_URL}/reset-password?token=${resetToken}`;
    
    const mailOptions = {
      from: process.env.EMAIL_USER,
      to: email,
      subject: 'Password Reset - Vocal Emotion AI',
      html: `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
          <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="color: #3b82f6; margin: 0;">Vocal Emotion AI</h1>
            <p style="color: #6b7280; margin: 5px 0 0 0;">AI-Powered Speech Emotion Recognition</p>
          </div>
          
          <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 10px; text-align: center; margin-bottom: 30px;">
            <h2 style="color: white; margin: 0 0 15px 0; font-size: 24px;">Password Reset Request</h2>
            <p style="color: white; margin: 0; font-size: 16px;">Hi ${firstName}, we received a request to reset your password</p>
          </div>
          
          <div style="background: #f8fafc; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 30px;">
            <p style="color: #374151; margin: 0 0 20px 0; font-size: 16px;">Click the button below to reset your password:</p>
            <a href="${resetUrl}" style="background: #3b82f6; color: white; padding: 12px 30px; text-decoration: none; border-radius: 6px; display: inline-block; font-weight: bold;">Reset Password</a>
            <p style="color: #6b7280; margin: 15px 0 0 0; font-size: 14px;">This link will expire in 1 hour</p>
          </div>
          
          <div style="background: #fef3c7; padding: 15px; border-radius: 8px; border-left: 4px solid #f59e0b;">
            <p style="color: #92400e; margin: 0; font-size: 14px;">
              <strong>Security Note:</strong> If you didn't request this password reset, please ignore this email.
            </p>
          </div>
        </div>
      `
    };

    const result = await transporter.sendMail(mailOptions);
    console.log('Password reset email sent successfully:', result.messageId);
    return { success: true, messageId: result.messageId };
  } catch (error) {
    console.error('Error sending password reset email:', error);
    return { success: false, error: error.message };
  }
};

module.exports = {
  sendOTPEmail,
  sendPasswordResetEmail
};
