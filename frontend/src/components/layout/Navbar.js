import React, { useEffect, useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useAuth } from '../../contexts/AuthContext';
import { useTheme } from '../../contexts/ThemeContext';
import ThemeToggle from '../ui/ThemeToggle';
import { 
  Menu, 
  X, 
  Brain, 
  User, 
  LogOut, 
  Settings, 
  BarChart3,
  MessageCircle,
  Mic,
  ChevronDown,
  Sparkles
} from 'lucide-react';

const Navbar = () => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const { user, logout, isAuthenticated } = useAuth();
  const { isDarkMode } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const handleLogout = () => {
    logout();
    navigate('/');
    setIsProfileOpen(false);
  };

  const isActive = (path) => location.pathname === path;

  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: BarChart3 },
    { name: 'Chat', path: '/chat', icon: MessageCircle },
    { name: 'Emotion Analysis', path: '/emotion-analysis', icon: Mic },
  ];

  return (
    <motion.nav 
      className={`${scrolled ? 'bg-white/80 dark:bg-dark-800/80 backdrop-blur-xl shadow-xl' : 'bg-white/90 dark:bg-dark-800/90 backdrop-blur-xl'} transition-all duration-300 sticky top-0 z-50 border-b border-white/20 dark:border-dark-700/50`}
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6 }}
    > 
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <motion.div 
            className="flex items-center"
            whileHover={{ scale: 1.05 }}
            transition={{ duration: 0.2 }}
          >
            <Link to="/" className="flex items-center space-x-3 group">
              <motion.div 
                className="h-10 w-10 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300"
                whileHover={{ rotate: 5 }}
                transition={{ duration: 0.2 }}
              >
                <Brain className="h-6 w-6 text-white" />
              </motion.div>
              <div className="flex flex-col">
                <span className="text-xl font-bold text-gray-900 dark:text-white group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors duration-300">
                  Vocal Emotion AI
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400 font-medium">
                  AI-Powered Analysis
                </span>
              </div>
            </Link>
          </motion.div>

          {/* Desktop Navigation */}
          {isAuthenticated && (
            <div className="hidden md:flex items-center space-x-2">
              {navItems.map((item, index) => {
                const Icon = item.icon;
                return (
                  <motion.div
                    key={item.name}
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3, delay: index * 0.1 }}
                  >
                    <Link
                      to={item.path}
                      className={`px-4 py-2 rounded-xl text-sm font-semibold transition-all duration-300 flex items-center space-x-2 group relative overflow-hidden ${
                        isActive(item.path)
                          ? 'bg-gradient-to-r from-primary-500 to-secondary-500 text-white shadow-lg shadow-primary-500/25'
                          : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-white/50 dark:hover:bg-dark-700/50 hover:shadow-lg'
                      }`}
                    >
                      {isActive(item.path) && (
                        <motion.div
                          className="absolute inset-0 bg-gradient-to-r from-primary-500 to-secondary-500"
                          layoutId="activeTab"
                          transition={{ duration: 0.3 }}
                        />
                      )}
                      <Icon className={`h-4 w-4 relative z-10 ${isActive(item.path) ? 'text-white' : 'group-hover:scale-110 transition-transform duration-200'}`} />
                      <span className="relative z-10">{item.name}</span>
                      {isActive(item.path) && (
                        <motion.div
                          className="absolute -top-1 -right-1 w-3 h-3 bg-accent-500 rounded-full"
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ duration: 2, repeat: Infinity }}
                        />
                      )}
                    </Link>
                  </motion.div>
                );
              })}
            </div>
          )}

          {/* Right side - Auth buttons or User menu */}
          <div className="flex items-center space-x-4">
            {isAuthenticated ? (
              <div className="relative">
                <motion.button
                  onClick={() => setIsProfileOpen(!isProfileOpen)}
                  className="flex items-center space-x-3 p-2 rounded-xl hover:bg-white/50 dark:hover:bg-dark-700/50 transition-all duration-300 group"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  <motion.div 
                    className="h-10 w-10 bg-gradient-to-br from-primary-500 to-secondary-500 rounded-2xl flex items-center justify-center shadow-lg group-hover:shadow-xl transition-all duration-300"
                    whileHover={{ rotate: 5 }}
                    transition={{ duration: 0.2 }}
                  >
                    {user?.profilePicture ? (
                      <img
                        src={user.profilePicture}
                        alt={user.firstName}
                        className="h-10 w-10 rounded-2xl object-cover"
                      />
                    ) : (
                      <User className="h-5 w-5 text-white" />
                    )}
                  </motion.div>
                  <div className="hidden sm:block text-left">
                    <div className="text-sm font-semibold text-gray-900 dark:text-white">
                      {user?.firstName || user?.username}
                    </div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Premium User
                    </div>
                  </div>
                  <motion.div
                    animate={{ rotate: isProfileOpen ? 180 : 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <ChevronDown className="h-4 w-4 text-gray-500 dark:text-gray-400" />
                  </motion.div>
                </motion.button>

                {/* Profile Dropdown */}
                <AnimatePresence>
                  {isProfileOpen && (
                    <motion.div 
                      className="absolute right-0 mt-2 w-56 bg-white/90 dark:bg-dark-800/90 backdrop-blur-xl rounded-2xl shadow-2xl border border-white/20 dark:border-dark-700/50 py-3 z-50"
                      initial={{ opacity: 0, y: -10, scale: 0.95 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={{ opacity: 0, y: -10, scale: 0.95 }}
                      transition={{ duration: 0.2 }}
                    >
                      <div className="px-4 py-2 border-b border-gray-200 dark:border-dark-700">
                        <div className="text-sm font-semibold text-gray-900 dark:text-white">
                          {user?.firstName || user?.username}
                        </div>
                        <div className="text-xs text-gray-500 dark:text-gray-400">
                          {user?.email}
                        </div>
                      </div>
                      
                      <Link
                        to="/profile"
                        className="flex px-4 py-3 text-sm text-gray-700 dark:text-gray-300 hover:bg-white/50 dark:hover:bg-dark-700/50 transition-all duration-200 items-center space-x-3 group"
                        onClick={() => setIsProfileOpen(false)}
                      >
                        <User className="h-4 w-4 group-hover:scale-110 transition-transform duration-200" />
                        <span>Profile</span>
                      </Link>
                      
                      <Link
                        to="/profile"
                        className="flex px-4 py-3 text-sm text-gray-700 dark:text-gray-300 hover:bg-white/50 dark:hover:bg-dark-700/50 transition-all duration-200 items-center space-x-3 group"
                        onClick={() => setIsProfileOpen(false)}
                      >
                        <Settings className="h-4 w-4 group-hover:scale-110 transition-transform duration-200" />
                        <span>Settings</span>
                      </Link>
                      
                      <div className="border-t border-gray-200 dark:border-dark-700 my-2"></div>
                      
                      <button
                        onClick={handleLogout}
                        className="w-full text-left px-4 py-3 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-all duration-200 flex items-center space-x-3 group"
                      >
                        <LogOut className="h-4 w-4 group-hover:scale-110 transition-transform duration-200" />
                        <span>Sign out</span>
                      </button>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            ) : (
              <div className="hidden sm:flex items-center space-x-3">
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: 0.1 }}
                >
                  <Link
                    to="/login"
                    className="btn-outline px-6 py-2 text-sm font-semibold"
                  >
                    Sign In
                  </Link>
                </motion.div>
                <motion.div
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: 0.2 }}
                >
                  <Link
                    to="/register"
                    className="btn-primary px-6 py-2 text-sm font-semibold flex items-center space-x-2"
                  >
                    <span>Get Started</span>
                    <Sparkles className="h-4 w-4" />
                  </Link>
                </motion.div>
              </div>
            )}

            {/* Theme Toggle */}
            <motion.div
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: 0.3 }}
            >
              <ThemeToggle />
            </motion.div>

            {/* Mobile menu button */}
            <motion.button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden p-3 rounded-xl hover:bg-white/50 dark:hover:bg-dark-700/50 transition-all duration-300"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <motion.div
                animate={{ rotate: isMenuOpen ? 90 : 0 }}
                transition={{ duration: 0.2 }}
              >
                {isMenuOpen ? (
                  <X className="h-6 w-6 text-gray-600 dark:text-gray-300" />
                ) : (
                  <Menu className="h-6 w-6 text-gray-600 dark:text-gray-300" />
                )}
              </motion.div>
            </motion.button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <AnimatePresence>
        {isMenuOpen && (
          <motion.div 
            className="md:hidden bg-white/90 dark:bg-dark-800/90 backdrop-blur-xl border-t border-white/20 dark:border-dark-700/50"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <div className="px-4 pt-4 pb-6 space-y-2">
              {isAuthenticated ? (
                <>
                  {navItems.map((item, index) => {
                    const Icon = item.icon;
                    return (
                      <motion.div
                        key={item.name}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ duration: 0.3, delay: index * 0.1 }}
                      >
                        <Link
                          to={item.path}
                          className={`flex px-4 py-3 rounded-xl text-base font-semibold transition-all duration-300 items-center space-x-3 group ${
                            isActive(item.path)
                              ? 'bg-gradient-to-r from-primary-500 to-secondary-500 text-white shadow-lg'
                              : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-white/50 dark:hover:bg-dark-700/50'
                          }`}
                          onClick={() => setIsMenuOpen(false)}
                        >
                          <Icon className={`h-5 w-5 ${isActive(item.path) ? 'text-white' : 'group-hover:scale-110 transition-transform duration-200'}`} />
                          <span>{item.name}</span>
                          {isActive(item.path) && (
                            <motion.div
                              className="ml-auto w-2 h-2 bg-white rounded-full"
                              animate={{ scale: [1, 1.2, 1] }}
                              transition={{ duration: 2, repeat: Infinity }}
                            />
                          )}
                        </Link>
                      </motion.div>
                    );
                  })}
                  
                  <div className="border-t border-gray-200 dark:border-dark-700 my-4"></div>
                  
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.4 }}
                  >
                    <Link
                      to="/profile"
                      className="flex px-4 py-3 rounded-xl text-base font-semibold text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-white/50 dark:hover:bg-dark-700/50 transition-all duration-300 items-center space-x-3 group"
                      onClick={() => setIsMenuOpen(false)}
                    >
                      <User className="h-5 w-5 group-hover:scale-110 transition-transform duration-200" />
                      <span>Profile</span>
                    </Link>
                  </motion.div>
                  
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.5 }}
                  >
                    <button
                      onClick={handleLogout}
                      className="w-full text-left px-4 py-3 rounded-xl text-base font-semibold text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 transition-all duration-300 flex items-center space-x-3 group"
                    >
                      <LogOut className="h-5 w-5 group-hover:scale-110 transition-transform duration-200" />
                      <span>Sign out</span>
                    </button>
                  </motion.div>
                </>
              ) : (
                <>
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.1 }}
                  >
                    <Link
                      to="/login"
                      className="block px-4 py-3 rounded-xl text-base font-semibold text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-white/50 dark:hover:bg-dark-700/50 transition-all duration-300"
                      onClick={() => setIsMenuOpen(false)}
                    >
                      Sign In
                    </Link>
                  </motion.div>
                  
                  <motion.div
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.2 }}
                  >
                    <Link
                      to="/register"
                      className="flex px-4 py-3 rounded-xl text-base font-semibold text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-white/50 dark:hover:bg-dark-700/50 transition-all duration-300 items-center space-x-2"
                      onClick={() => setIsMenuOpen(false)}
                    >
                      <span>Get Started</span>
                      <Sparkles className="h-4 w-4" />
                    </Link>
                  </motion.div>
                </>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
};

export default Navbar;
