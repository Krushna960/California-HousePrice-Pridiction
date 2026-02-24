// Main JavaScript for California House Price Prediction App

(function() {
  'use strict';

  // Form submission handling
  const form = document.getElementById('predictForm');
  const submitBtn = document.getElementById('submitBtn');
  const btnText = document.getElementById('btnText');
  const btnSpinner = document.getElementById('btnSpinner');

  if (form && submitBtn) {
    form.addEventListener('submit', function(e) {
      // Validate form before submission
      if (!form.checkValidity()) {
        e.preventDefault();
        e.stopPropagation();
        form.classList.add('was-validated');
        return;
      }

      // Show loading state
      submitBtn.disabled = true;
      if (btnText) btnText.textContent = 'Processing...';
      if (btnSpinner) btnSpinner.style.display = 'inline-block';
      
      // Scroll to top to show result
      setTimeout(() => {
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }, 100);
    });

    // Real-time validation feedback
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
      input.addEventListener('blur', function() {
        if (this.checkValidity()) {
          this.style.borderColor = 'var(--success)';
        } else {
          this.style.borderColor = '';
        }
      });

      input.addEventListener('input', function() {
        if (this.style.borderColor === 'rgb(34, 197, 94)') {
          this.style.borderColor = '';
        }
      });
    });
  }

  // Reset form function (called from predict.html)
  window.resetForm = function() {
    if (form) {
      form.reset();
      form.classList.remove('was-validated');
      
      // Reset button state
      if (submitBtn) submitBtn.disabled = false;
      if (btnText) btnText.textContent = 'Get Price Estimate';
      if (btnSpinner) btnSpinner.style.display = 'none';
      
      // Scroll to form
      setTimeout(() => {
        form.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 100);
    }
  };

  // Smooth scroll for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href !== '#' && href.length > 1) {
        const target = document.querySelector(href);
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }
    });
  });

  // Add animation on scroll
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, observerOptions);

  // Observe elements for scroll animations
  document.querySelectorAll('.feature-card, .card, .tech-item').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
    observer.observe(el);
  });

  // Navbar scroll effect
  let lastScroll = 0;
  const navbar = document.querySelector('.navbar');
  
  if (navbar) {
    window.addEventListener('scroll', function() {
      const currentScroll = window.pageYOffset;
      
      if (currentScroll > 100) {
        navbar.style.background = 'rgba(26, 35, 50, 0.95)';
        navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.3)';
      } else {
        navbar.style.background = 'rgba(26, 35, 50, 0.8)';
        navbar.style.boxShadow = 'none';
      }
      
      lastScroll = currentScroll;
    });
  }

  // Add ripple effect to buttons
  document.querySelectorAll('.btn').forEach(button => {
    button.addEventListener('click', function(e) {
      const ripple = document.createElement('span');
      const rect = this.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      const x = e.clientX - rect.left - size / 2;
      const y = e.clientY - rect.top - size / 2;
      
      ripple.style.width = ripple.style.height = size + 'px';
      ripple.style.left = x + 'px';
      ripple.style.top = y + 'px';
      ripple.classList.add('ripple');
      
      this.appendChild(ripple);
      
      setTimeout(() => {
        ripple.remove();
      }, 600);
    });
  });

  // Add CSS for ripple effect
  const style = document.createElement('style');
  style.textContent = `
    .btn {
      position: relative;
      overflow: hidden;
    }
    .ripple {
      position: absolute;
      border-radius: 50%;
      background: rgba(255, 255, 255, 0.3);
      transform: scale(0);
      animation: ripple-animation 0.6s ease-out;
      pointer-events: none;
    }
    @keyframes ripple-animation {
      to {
        transform: scale(4);
        opacity: 0;
      }
    }
  `;
  document.head.appendChild(style);

  // Number input formatting
  const numberInputs = document.querySelectorAll('input[type="number"], input[type="text"][name*="income"], input[type="text"][name*="longitude"], input[type="text"][name*="latitude"]');
  numberInputs.forEach(input => {
    input.addEventListener('input', function() {
      // Remove any non-numeric characters except decimal point and minus
      if (this.name === 'median_income' || this.name === 'longitude' || this.name === 'latitude') {
        this.value = this.value.replace(/[^0-9.\-]/g, '');
      }
    });
  });

  // Console welcome message
  console.log('%c🏠 California House Price Prediction', 'font-size: 20px; font-weight: bold; color: #3b82f6;');
  console.log('%cPowered by Machine Learning', 'font-size: 12px; color: #8b9cb3;');

})();
