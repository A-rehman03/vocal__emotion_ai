import React from 'react';
import { Link } from 'react-router-dom';
import { Brain, Heart, Mail, Twitter, Github, Linkedin } from 'lucide-react';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="bg-gray-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Company Info */}
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center space-x-2 mb-4">
              <div className="h-8 w-8 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-lg flex items-center justify-center">
                <Brain className="h-5 w-5 text-white" />
              </div>
              <span className="text-xl font-bold">Vocal Emotion AI</span>
            </div>
            <p className="text-gray-300 mb-6 max-w-md">
              Revolutionizing human-computer interaction with AI-driven speech emotion recognition. 
              Experience emotionally intelligent conversations that understand and respond to your feelings.
            </p>
            <div className="flex space-x-4">
              <a
                href="https://twitter.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                <Twitter className="h-5 w-5" />
              </a>
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                <Github className="h-5 w-5" />
              </a>
              <a
                href="https://linkedin.com"
                target="_blank"
                rel="noopener noreferrer"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                <Linkedin className="h-5 w-5" />
              </a>
              <a
                href="mailto:contact@vocalemotionai.com"
                className="text-gray-400 hover:text-white transition-colors duration-200"
              >
                <Mail className="h-5 w-5" />
              </a>
            </div>
          </div>

          {/* Product Links */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Product</h3>
            <ul className="space-y-2">
              <li>
                <Link
                  to="/features"
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  Features
                </Link>
              </li>
              <li>
                <Link
                  to="/pricing"
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  Pricing
                </Link>
              </li>
              <li>
                <Link
                  to="/api"
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  API
                </Link>
              </li>
              <li>
                <Link
                  to="/integrations"
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  Integrations
                </Link>
              </li>
            </ul>
          </div>

          {/* Company Links */}
          <div>
            <h3 className="text-lg font-semibold mb-4">Company</h3>
            <ul className="space-y-2">
              <li>
                <Link
                  to="/about"
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  About
                </Link>
              </li>
              <li>
                <Link
                  to="/blog"
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  Blog
                </Link>
              </li>
              <li>
                <Link
                  to="/careers"
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  Careers
                </Link>
              </li>
              <li>
                <Link
                  to="/contact"
                  className="text-gray-300 hover:text-white transition-colors duration-200"
                >
                  Contact
                </Link>
              </li>
            </ul>
          </div>
        </div>

        {/* Bottom Section */}
        <div className="border-t border-gray-800 mt-12 pt-8">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-2 text-gray-400 mb-4 md:mb-0">
              <Heart className="h-4 w-4 text-red-400" />
              <span className="text-sm">
                Made with emotion recognition technology
              </span>
            </div>
            <div className="flex flex-col md:flex-row items-center space-y-2 md:space-y-0 md:space-x-6 text-sm text-gray-400">
              <Link
                to="/privacy"
                className="hover:text-white transition-colors duration-200"
              >
                Privacy Policy
              </Link>
              <Link
                to="/terms"
                className="hover:text-white transition-colors duration-200"
              >
                Terms of Service
              </Link>
              <span>&copy; {currentYear} Vocal Emotion AI. All rights reserved.</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
