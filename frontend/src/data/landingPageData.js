import {
    Brain,
    Heart,
    Mic,
    MessageCircle,
    Zap,
    Shield
} from 'lucide-react';

export const features = [
    {
        icon: Brain,
        title: 'AI-Powered Emotion Recognition',
        description: 'Advanced machine learning algorithms that accurately detect emotions from speech patterns, tone, and voice characteristics.',
        color: 'primary'
    },
    {
        icon: Heart,
        title: 'Emotionally Intelligent Responses',
        description: 'Get personalized responses that understand and adapt to your emotional state, creating more meaningful interactions.',
        color: 'secondary'
    },
    {
        icon: Mic,
        title: 'Real-Time Voice Analysis',
        description: 'Instant emotion detection from live voice input with minimal latency for seamless conversation flow.',
        color: 'emotion-excited'
    },
    {
        icon: MessageCircle,
        title: 'Natural Conversation Flow',
        description: 'Engage in natural, context-aware conversations that feel human and emotionally responsive.',
        color: 'emotion-calm'
    },
    {
        icon: Zap,
        title: 'Lightning Fast Processing',
        description: 'Optimized algorithms ensure quick response times while maintaining high accuracy in emotion detection.',
        color: 'emotion-happy'
    },
    {
        icon: Shield,
        title: 'Privacy & Security First',
        description: 'Your voice data is protected with enterprise-grade security and privacy controls.',
        color: 'emotion-confident'
    }
];

export const stats = [
    { number: '95%', label: 'Precision Detection' },
    { number: '<100ms', label: 'Ultra-Fast Response' },
    { number: '7+', label: 'Emotion Spectrum' },
    { number: '24/7', label: 'Continuous Availability' }
];

export const testimonials = [
    {
        name: "Sarah Johnson",
        role: "Mental Health Professional",
        content: "This tool has revolutionized how I track my patients' emotional progress. The accuracy is astounding.",
        rating: 5
    },
    {
        name: "David Chen",
        role: "User Experience Researcher",
        content: "Vocal Emotion AI provides insights that traditional surveys simply cannot capture. A game-changer for UX testing.",
        rating: 5
    },
    {
        name: "Emily Rodriguez",
        role: "Customer Success Manager",
        content: "Understanding customer sentiment in real-time has helped us improve our satisfaction scores by 40%.",
        rating: 5
    }
];

export const howItWorksSteps = [
    {
        step: "1",
        title: "Speak Naturally",
        description: "Simply speak into your microphone or upload an audio file. Our system works with any voice input.",
        color: "primary",
        icon: Mic
    },
    {
        step: "2",
        title: "AI Analysis",
        description: "Our advanced AI analyzes your speech patterns, tone, and voice characteristics in real-time.",
        color: "secondary",
        icon: Brain
    },
    {
        step: "3",
        title: "Get Response",
        description: "Receive emotionally intelligent responses that understand and adapt to your emotional state.",
        color: "accent",
        icon: Heart
    }
];
