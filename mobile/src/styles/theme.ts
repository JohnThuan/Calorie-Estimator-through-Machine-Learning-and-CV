/**
 * Calorie Estimation through CV - Professional Theme
 * Research-grade color palette and design tokens
 */

export const theme = {
  colors: {
    // Primary Colors - Academic Blue
    primary: '#2196F3',
    primaryDark: '#0A1F44',
    primaryLight: '#E3F2FD',
    
    // Accent Colors
    accent: '#2196F3',
    accentLight: '#B3E5FC',
    
    // Background
    background: '#F8F9FA',
    surface: '#FFFFFF',
    surfaceAlt: '#FAFAFA',
    
    // Text
    text: '#0A1F44',
    textSecondary: '#546E7A',
    textTertiary: '#78909C',
    textLight: '#B0BEC5',
    
    // Status Colors
    success: '#4CAF50',
    warning: '#FF9800',
    error: '#F44336',
    info: '#2196F3',
    
    // Confidence Levels
    highConfidence: '#4CAF50',
    mediumConfidence: '#FF9800',
    lowConfidence: '#F44336',
    
    // Nutrients
    protein: '#2196F3',
    carbs: '#4CAF50',
    fat: '#FF9800',
    
    // Borders & Dividers
    border: '#CFD8DC',
    divider: '#ECEFF1',
    
    // Overlays
    overlay: 'rgba(10, 31, 68, 0.9)',
    overlayLight: 'rgba(10, 31, 68, 0.7)',
  },
  
  spacing: {
    xs: 4,
    sm: 8,
    md: 12,
    lg: 16,
    xl: 20,
    xxl: 24,
  },
  
  borderRadius: {
    sm: 4,
    md: 6,
    lg: 8,
    xl: 12,
    round: 999,
  },
  
  typography: {
    // Headers
    h1: {
      fontSize: 32,
      fontWeight: '700',
      letterSpacing: 0.5,
    },
    h2: {
      fontSize: 24,
      fontWeight: '700',
      letterSpacing: 0.5,
    },
    h3: {
      fontSize: 20,
      fontWeight: '600',
      letterSpacing: 0.5,
    },
    h4: {
      fontSize: 18,
      fontWeight: '600',
      letterSpacing: 0.5,
    },
    
    // Body
    body: {
      fontSize: 16,
      fontWeight: '400',
      lineHeight: 24,
    },
    bodySmall: {
      fontSize: 14,
      fontWeight: '400',
      lineHeight: 20,
    },
    
    // Caption
    caption: {
      fontSize: 12,
      fontWeight: '400',
      lineHeight: 16,
    },
    
    // Labels
    label: {
      fontSize: 14,
      fontWeight: '500',
      letterSpacing: 0.5,
    },
    labelSmall: {
      fontSize: 12,
      fontWeight: '500',
      letterSpacing: 0.5,
    },
  },
  
  elevation: {
    none: 0,
    low: 1,
    medium: 2,
    high: 4,
  },
};

export type Theme = typeof theme;




