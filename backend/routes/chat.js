const express = require('express');
const { body, validationResult } = require('express-validator');
const { protect } = require('../middleware/auth');

const router = express.Router();

// @route   POST /api/chat/message
// @desc    Send a message to the AI chatbot
// @access  Private
router.post('/message', protect, [
  body('message')
    .notEmpty()
    .withMessage('Message is required')
    .isLength({ max: 1000 })
    .withMessage('Message cannot exceed 1000 characters'),
  body('emotion')
    .optional()
    .isIn(['happy', 'sad', 'angry', 'neutral', 'excited', 'calm', 'anxious', 'confident'])
    .withMessage('Invalid emotion value'),
  body('context')
    .optional()
    .isObject()
    .withMessage('Context must be an object')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ 
        message: 'Validation failed',
        errors: errors.array() 
      });
    }

    const { message, emotion, context } = req.body;

    // Here you would integrate with OpenAI or other AI service
    // This is a placeholder response that considers emotion
    
    let aiResponse = '';
    let responseEmotion = 'neutral';
    
    if (emotion) {
      // Emotion-aware responses
      switch (emotion) {
        case 'happy':
          aiResponse = "I'm so glad you're feeling happy! Your positive energy is wonderful. What's bringing you joy today?";
          responseEmotion = 'excited';
          break;
        case 'sad':
          aiResponse = "I can sense that you're going through a difficult time. It's okay to feel sad, and I'm here to listen. Would you like to talk about what's on your mind?";
          responseEmotion = 'calm';
          break;
        case 'angry':
          aiResponse = "I understand you're feeling frustrated. Let's take a moment to breathe and work through this together. What happened?";
          responseEmotion = 'calm';
          break;
        case 'excited':
          aiResponse = "Your excitement is contagious! I'd love to hear more about what's got you so energized. This sounds amazing!";
          responseEmotion = 'excited';
          break;
        case 'anxious':
          aiResponse = "I can hear the worry in your voice. Let's take this one step at a time. You're not alone, and we'll figure this out together.";
          responseEmotion = 'calm';
          break;
        case 'confident':
          aiResponse = "I love your confidence! You sound like you're in control and ready to take on the world. What's your next move?";
          responseEmotion = 'excited';
          break;
        default:
          aiResponse = "Thank you for sharing that with me. I'm here to listen and help however I can. How are you feeling about this?";
      }
    } else {
      // General response
      aiResponse = "Thank you for your message. I'm here to help and support you. How can I assist you today?";
    }

    // Mock conversation context
    const conversationId = `conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    res.json({
      message: 'Message processed successfully',
      conversationId,
      userMessage: {
        text: message,
        emotion: emotion || 'neutral',
        timestamp: new Date().toISOString()
      },
      aiResponse: {
        text: aiResponse,
        emotion: responseEmotion,
        timestamp: new Date().toISOString()
      },
      context: context || {}
    });

  } catch (error) {
    console.error('Chat message error:', error);
    res.status(500).json({ message: 'Server error while processing message' });
  }
});

// @route   POST /api/chat/voice
// @desc    Send voice message to the AI chatbot
// @access  Private
router.post('/voice', protect, async (req, res) => {
  try {
    const { audioData, emotion, context, conversationId } = req.body;

    if (!audioData) {
      return res.status(400).json({ message: 'Audio data is required' });
    }

    // Process the voice message through emotion analysis
    try {
      // Convert base64 audio data to buffer
      const audioBuffer = Buffer.from(audioData, 'base64');
      
      // Send to emotion analysis service
      const formData = new FormData();
      formData.append('audio', audioBuffer, {
        filename: 'voice-message.wav',
        contentType: 'audio/wav'
      });

      const emotionResponse = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        body: formData
      });

      let detectedEmotion = 'neutral';
      let confidence = 0.5;
      let transcription = "Voice message received";

      if (emotionResponse.ok) {
        const emotionData = await emotionResponse.json();
        detectedEmotion = emotionData.labels?.top1 || 'neutral';
        confidence = emotionData.labels?.confidence || 0.5;
        transcription = `Voice message (${detectedEmotion})`;
      }

      // Generate contextual AI response based on detected emotion
      let aiResponse = '';
      switch (detectedEmotion) {
        case 'happy':
          aiResponse = "I can hear the joy in your voice! That's wonderful to hear. What's making you feel so happy today?";
          break;
        case 'sad':
          aiResponse = "I sense some sadness in your tone. It's okay to feel this way. Would you like to talk about what's on your mind?";
          break;
        case 'angry':
          aiResponse = "I notice some frustration in your voice. Let's take a moment to breathe and work through this together. What's bothering you?";
          break;
        case 'fear':
          aiResponse = "I can hear some worry in your voice. You're not alone, and we can work through this together. What's concerning you?";
          break;
        case 'surprise':
          aiResponse = "You sound surprised! I'd love to hear more about what caught you off guard.";
          break;
        case 'disgust':
          aiResponse = "I can sense some strong feelings in your voice. What's on your mind right now?";
          break;
        case 'neutral':
          aiResponse = "I'm here to listen and help. How are you feeling today?";
          break;
        default:
          aiResponse = "Thank you for your voice message. I'm here to listen and help however I can.";
      }

      const currentConversationId = conversationId || `voice_conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      res.json({
        message: 'Voice message processed successfully',
        conversationId: currentConversationId,
        transcription,
        detectedEmotion,
        confidence,
        aiResponse,
        timestamp: new Date().toISOString(),
        context: context || {}
      });

    } catch (processingError) {
      console.error('Voice processing error:', processingError);
      
      // Fallback response
      const fallbackEmotion = emotion || 'neutral';
      const fallbackResponse = "I received your voice message. I'm here to listen and help. How are you feeling today?";
      
      res.json({
        message: 'Voice message processed (fallback mode)',
        conversationId: conversationId || `voice_conv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        transcription: "Voice message received",
        detectedEmotion: fallbackEmotion,
        confidence: 0.5,
        aiResponse: fallbackResponse,
        timestamp: new Date().toISOString(),
        note: 'Using fallback processing - emotion analysis unavailable'
      });
    }

  } catch (error) {
    console.error('Voice chat error:', error);
    res.status(500).json({ message: 'Server error while processing voice message' });
  }
});

// @route   GET /api/chat/history
// @desc    Get chat conversation history
// @access  Private
router.get('/history', protect, async (req, res) => {
  try {
    const { page = 1, limit = 20, conversationId } = req.query;

    // Here you would fetch from a database
    // This is a placeholder response
    
    const mockConversations = conversationId ? 
      // Single conversation
      [{
        id: conversationId,
        messages: [
          {
            id: 'msg_1',
            type: 'user',
            text: 'Hello, how are you today?',
            emotion: 'neutral',
            timestamp: new Date(Date.now() - 3600000).toISOString()
          },
          {
            id: 'msg_2',
            type: 'ai',
            text: 'Hello! I\'m doing well, thank you for asking. How are you feeling today?',
            emotion: 'happy',
            timestamp: new Date(Date.now() - 3500000).toISOString()
          }
        ]
      }] :
      // Multiple conversations
      Array.from({ length: Math.min(limit, 5) }, (_, i) => ({
        id: `conv_${Date.now()}_${i}`,
        title: `Conversation ${i + 1}`,
        lastMessage: `Last message from conversation ${i + 1}`,
        messageCount: Math.floor(Math.random() * 20) + 1,
        lastActivity: new Date(Date.now() - i * 86400000).toISOString()
      }));

    res.json({
      message: 'Chat history retrieved successfully',
      conversations: mockConversations,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total: 15, // Mock total
        pages: Math.ceil(15 / limit)
      }
    });

  } catch (error) {
    console.error('Get chat history error:', error);
    res.status(500).json({ message: 'Server error while fetching chat history' });
  }
});

// @route   DELETE /api/chat/conversation/:id
// @desc    Delete a conversation
// @access  Private
router.delete('/conversation/:id', protect, async (req, res) => {
  try {
    const { id } = req.params;

    // Here you would delete from database
    // This is a placeholder response

    res.json({
      message: 'Conversation deleted successfully',
      conversationId: id
    });

  } catch (error) {
    console.error('Delete conversation error:', error);
    res.status(500).json({ message: 'Server error while deleting conversation' });
  }
});

// @route   POST /api/chat/feedback
// @desc    Submit feedback on AI responses
// @access  Private
router.post('/feedback', protect, [
  body('conversationId')
    .notEmpty()
    .withMessage('Conversation ID is required'),
  body('messageId')
    .notEmpty()
    .withMessage('Message ID is required'),
  body('rating')
    .isInt({ min: 1, max: 5 })
    .withMessage('Rating must be between 1 and 5'),
  body('feedback')
    .optional()
    .isString()
    .isLength({ max: 500 })
    .withMessage('Feedback cannot exceed 500 characters')
], async (req, res) => {
  try {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
      return res.status(400).json({ 
        message: 'Validation failed',
        errors: errors.array() 
      });
    }

    const { conversationId, messageId, rating, feedback } = req.body;

    // Here you would save feedback to improve the AI model
    // This is a placeholder response

    res.json({
      message: 'Feedback submitted successfully',
      feedback: {
        conversationId,
        messageId,
        rating,
        feedback: feedback || 'no_additional_feedback',
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Submit chat feedback error:', error);
    res.status(500).json({ message: 'Server error while submitting feedback' });
  }
});

module.exports = router;
