const express = require('express');
const multer = require('multer');
const { protect } = require('../middleware/auth');

const router = express.Router();

// Configure multer for audio file uploads
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
  limits: {
    fileSize: parseInt(process.env.MAX_FILE_SIZE) || 10 * 1024 * 1024, // 10MB default
  },
  fileFilter: (req, file, cb) => {
    // Accept audio files
    if (file.mimetype.startsWith('audio/')) {
      cb(null, true);
    } else {
      cb(new Error('Only audio files are allowed'), false);
    }
  }
});

// @route   POST /api/emotion/analyze
// @desc    Analyze speech emotion from audio file
// @access  Private
router.post('/analyze', protect, upload.single('audio'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ message: 'No audio file provided' });
    }

    const audioBuffer = req.file.buffer;
    const audioType = req.file.mimetype;

    // Send audio to Python emotion analysis service
    try {
      const formData = new FormData();
      const blob = new Blob([audioBuffer], { type: audioType });
      formData.append('audio', blob, req.file.originalname || 'audio.wav');

      const pythonResponse = await fetch('http://localhost:5002/analyze', {
        method: 'POST',
        body: formData
      });

      if (!pythonResponse.ok) {
        throw new Error(`Python service error: ${pythonResponse.status}`);
      }

      const pythonData = await pythonResponse.json();

      if (pythonData.error) {
        throw new Error(pythonData.error);
      }

      // Extract emotion from Python response
      // New format: { predictions: [{ label: 'happy', score: 0.9 }, ...] }
      const topPrediction = pythonData.predictions && pythonData.predictions.length > 0
        ? pythonData.predictions[0]
        : { label: 'neutral', score: 0.0 };

      const detectedEmotion = topPrediction.label;
      const confidence = topPrediction.score;

      // Generate contextual response based on detected emotion
      let response = '';
      switch (detectedEmotion.toLowerCase()) {
        case 'happy':
          response = "I can hear the joy in your voice! That's wonderful to hear. What's making you feel so happy today?";
          break;
        case 'sad':
          response = "I sense some sadness in your tone. It's okay to feel this way. Would you like to talk about what's on your mind?";
          break;
        case 'angry':
          response = "I notice some frustration in your voice. Let's take a moment to breathe and work through this together. What's bothering you?";
          break;
        case 'fear':
        case 'fearful':
          response = "I can hear some worry in your voice. You're not alone, and we can work through this together. What's concerning you?";
          break;
        case 'surprise':
        case 'surprised':
          response = "You sound surprised! I'd love to hear more about what caught you off guard.";
          break;
        case 'disgust':
          response = "I can sense some strong feelings in your voice. What's on your mind right now?";
          break;
        case 'neutral':
          response = "I'm here to listen and help. How are you feeling today?";
          break;
        case 'calm':
          response = "You sound very calm and centered. That's a wonderful state to be in.";
          break;
        default:
          response = "Thank you for sharing that with me. I'm here to listen and help however I can.";
      }

      // Map probabilities for frontend
      const allProbabilities = {};
      if (pythonData.predictions) {
        pythonData.predictions.forEach(p => {
          allProbabilities[p.label] = p.score;
        });
      }

      res.json({
        message: 'Emotion analysis completed',
        audioInfo: {
          originalName: req.file.originalname,
          size: req.file.size,
          mimetype: audioType,
          duration: 0 // New model doesn't return duration yet
        },
        analysis: {
          detectedEmotion,
          confidence: parseFloat(confidence),
          timestamp: new Date().toISOString(),
          allProbabilities: allProbabilities,
          method: 'transformer_model'
        },
        labels: {
          top1: detectedEmotion,
          probs: allProbabilities
        },
        input_shape: [1, 16000],
        features: { sequence_shape: [1, 128, 128] },
        response
      });

    } catch (pythonError) {
      console.error('Python service error:', pythonError);

      // Fallback to mock analysis if Python service is unavailable
      const emotions = ['happy', 'sad', 'angry', 'neutral', 'excited', 'calm'];
      const detectedEmotion = emotions[Math.floor(Math.random() * emotions.length)];
      const confidence = Math.random() * 0.4 + 0.6;

      let response = '';
      switch (detectedEmotion) {
        case 'happy':
          response = "I can hear the joy in your voice! That's wonderful to hear.";
          break;
        case 'sad':
          response = "I sense some sadness in your tone. Would you like to talk about what's on your mind?";
          break;
        case 'angry':
          response = "I notice some frustration in your voice. Let's work through this together.";
          break;
        case 'excited':
          response = "Your excitement is contagious! I'm excited to hear more about this.";
          break;
        case 'calm':
          response = "You sound very calm and centered. That's a wonderful state to be in.";
          break;
        default:
          response = "I'm here to listen and help. How are you feeling today?";
      }

      res.json({
        message: 'Emotion analysis completed (fallback mode)',
        audioInfo: {
          originalName: req.file.originalname,
          size: req.file.size,
          mimetype: audioType
        },
        analysis: {
          detectedEmotion,
          confidence: confidence.toFixed(2),
          timestamp: new Date().toISOString(),
          note: 'Using fallback analysis - Python service unavailable'
        },
        response
      });
    }

  } catch (error) {
    console.error('Emotion analysis error:', error);
    res.status(500).json({ message: 'Server error during emotion analysis' });
  }
});

// @route   POST /api/emotion/realtime
// @desc    Real-time emotion analysis (for streaming audio)
// @access  Private
router.post('/realtime', protect, async (req, res) => {
  try {
    const { audioChunk, sessionId } = req.body;

    if (!audioChunk) {
      return res.status(400).json({ message: 'No audio chunk provided' });
    }

    // Here you would implement real-time emotion analysis
    // This is a placeholder for streaming emotion detection

    // Mock real-time analysis
    const emotions = ['happy', 'sad', 'angry', 'neutral', 'excited', 'calm'];
    const detectedEmotion = emotions[Math.floor(Math.random() * emotions.length)];
    const confidence = Math.random() * 0.4 + 0.6;

    res.json({
      message: 'Real-time emotion analysis',
      sessionId,
      analysis: {
        detectedEmotion,
        confidence: confidence.toFixed(2),
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Real-time emotion analysis error:', error);
    res.status(500).json({ message: 'Server error during real-time analysis' });
  }
});

// @route   GET /api/emotion/history
// @desc    Get user's emotion analysis history
// @access  Private
router.get('/history', protect, async (req, res) => {
  try {
    const { page = 1, limit = 10 } = req.query;

    // Here you would fetch from a database
    // This is a placeholder response

    const mockHistory = Array.from({ length: Math.min(limit, 5) }, (_, i) => ({
      id: `analysis_${Date.now()}_${i}`,
      emotion: ['happy', 'sad', 'angry', 'neutral', 'excited', 'calm'][Math.floor(Math.random() * 6)],
      confidence: (Math.random() * 0.4 + 0.6).toFixed(2),
      timestamp: new Date(Date.now() - i * 86400000).toISOString(), // Mock dates
      audioDuration: Math.floor(Math.random() * 30) + 5 // 5-35 seconds
    }));

    res.json({
      message: 'Emotion analysis history retrieved',
      history: mockHistory,
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total: 25, // Mock total
        pages: Math.ceil(25 / limit)
      }
    });

  } catch (error) {
    console.error('Get emotion history error:', error);
    res.status(500).json({ message: 'Server error while fetching emotion history' });
  }
});

// @route   POST /api/emotion/feedback
// @desc    Submit feedback on emotion detection accuracy
// @access  Private
router.post('/feedback', protect, async (req, res) => {
  try {
    const { analysisId, actualEmotion, accuracy, feedback } = req.body;

    if (!analysisId || !actualEmotion) {
      return res.status(400).json({ message: 'Analysis ID and actual emotion are required' });
    }

    // Here you would save feedback to improve the AI model
    // This is a placeholder response

    res.json({
      message: 'Feedback submitted successfully',
      feedback: {
        analysisId,
        actualEmotion,
        accuracy: accuracy || 'not_provided',
        feedback: feedback || 'no_additional_feedback',
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    console.error('Submit feedback error:', error);
    res.status(500).json({ message: 'Server error while submitting feedback' });
  }
});

module.exports = router;
