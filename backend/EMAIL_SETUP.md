# Email Setup Guide

## Environment Variables Required

Add these to your `.env` file in the backend directory:

```env
# Email Configuration (Gmail)
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_gmail_app_password

# Frontend URL
FRONTEND_URL=http://localhost:3000
```

## Gmail Setup

1. **Enable 2-Factor Authentication** on your Gmail account
2. **Generate an App Password**:
   - Go to Google Account settings
   - Security → 2-Step Verification → App passwords
   - Generate a new app password for "Mail"
   - Use this password as `EMAIL_PASS` in your .env file

## Alternative Email Services

For production, consider using:
- **SendGrid** (recommended)
- **AWS SES**
- **Mailgun**
- **Nodemailer with custom SMTP**

## Testing Email

1. Set up the environment variables
2. Start the backend server
3. Register a new user
4. Check your email for the OTP code

## Troubleshooting

- **"Invalid login"**: Check your Gmail app password
- **"Less secure app access"**: Use app passwords instead
- **No emails received**: Check spam folder
- **Rate limiting**: Gmail has daily sending limits
