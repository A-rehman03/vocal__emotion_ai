import React, { useEffect, useState } from 'react';

const AnimatedBackground = ({ children }) => {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 10);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  return (
    <div className="relative min-h-screen">
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-32 -left-32 h-80 w-80 rounded-full blur-3xl opacity-30 animate-gradient bg-gradient-to-tr from-primary-300 to-secondary-300"></div>
        <div className="absolute -bottom-32 -right-32 h-96 w-96 rounded-full blur-3xl opacity-30 animate-gradient bg-gradient-to-tr from-secondary-200 to-primary-300" style={{ animationDelay: '1.5s' }}></div>
        <div className={`absolute inset-0 ${scrolled ? 'opacity-10' : 'opacity-20'} transition-opacity bg-grid-pattern`}></div>
      </div>
      <div className="relative">{children}</div>
    </div>
  );
};

export default AnimatedBackground;


