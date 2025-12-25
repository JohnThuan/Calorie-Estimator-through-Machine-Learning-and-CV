import React, { useEffect, useRef } from 'react';
import { View, StyleSheet, Animated, Dimensions } from 'react-native';
import { Text } from 'react-native-paper';
import Svg, { Circle, G, Line, Text as SvgText } from 'react-native-svg';

const { width } = Dimensions.get('window');
const GAUGE_SIZE = Math.min(width * 0.8, 300);
const RADIUS = GAUGE_SIZE / 2 - 20;
const STROKE_WIDTH = 20;

interface CalorieGaugeProps {
  value: number;
  min: number;
  max: number;
  confidence: number;
}

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

export default function CalorieGauge({ value, min, max, confidence }: CalorieGaugeProps) {
  const animatedValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.spring(animatedValue, {
      toValue: value,
      friction: 5,
      tension: 40,
      useNativeDriver: true,
    }).start();
  }, [value]);

  // Calculate gauge parameters
  const circumference = 2 * Math.PI * RADIUS;
  const maxCalories = 2500; // Reference max for gauge
  const percentage = Math.min(value / maxCalories, 1);
  const strokeDashoffset = circumference * (1 - percentage);

  // Confidence-based color
  const getColor = () => {
    if (confidence > 0.7) return '#4CAF50';
    if (confidence > 0.4) return '#FF9800';
    return '#F44336';
  };

  // Range indicator angles
  const minAngle = -135 + (270 * Math.min(min / maxCalories, 1));
  const maxAngle = -135 + (270 * Math.min(max / maxCalories, 1));

  return (
    <View style={styles.container}>
      <View style={styles.gaugeContainer}>
        <Svg width={GAUGE_SIZE} height={GAUGE_SIZE}>
          <G rotation="-135" origin={`${GAUGE_SIZE / 2}, ${GAUGE_SIZE / 2}`}>
            {/* Background circle */}
            <Circle
              cx={GAUGE_SIZE / 2}
              cy={GAUGE_SIZE / 2}
              r={RADIUS}
              stroke="#E0E0E0"
              strokeWidth={STROKE_WIDTH}
              fill="none"
              strokeDasharray={`${circumference * 0.75} ${circumference}`}
            />
            
            {/* Value circle */}
            <Circle
              cx={GAUGE_SIZE / 2}
              cy={GAUGE_SIZE / 2}
              r={RADIUS}
              stroke={getColor()}
              strokeWidth={STROKE_WIDTH}
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={circumference - circumference * 0.75 * percentage}
              strokeLinecap="round"
            />
          </G>

          {/* Center text */}
          <SvgText
            x={GAUGE_SIZE / 2}
            y={GAUGE_SIZE / 2 - 10}
            fontSize="36"
            fontWeight="bold"
            fill={getColor()}
            textAnchor="middle"
          >
            {Math.round(value)}
          </SvgText>
          <SvgText
            x={GAUGE_SIZE / 2}
            y={GAUGE_SIZE / 2 + 20}
            fontSize="16"
            fill="#666"
            textAnchor="middle"
          >
            kcal
          </SvgText>
        </Svg>

        {/* Confidence indicator */}
        <View style={styles.confidenceContainer}>
          <View style={[styles.confidenceBar, { width: `${confidence * 100}%`, backgroundColor: getColor() }]} />
        </View>
      </View>

      {/* Legend */}
      <View style={styles.legend}>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: '#E0E0E0' }]} />
          <Text style={styles.legendText}>Max reference (2500)</Text>
        </View>
        <View style={styles.legendItem}>
          <View style={[styles.legendDot, { backgroundColor: getColor() }]} />
          <Text style={styles.legendText}>Your meal estimate</Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    paddingVertical: 20,
  },
  gaugeContainer: {
    position: 'relative',
    alignItems: 'center',
    marginBottom: 20,
  },
  confidenceContainer: {
    width: GAUGE_SIZE * 0.6,
    height: 8,
    backgroundColor: '#E0E0E0',
    borderRadius: 4,
    marginTop: 16,
    overflow: 'hidden',
  },
  confidenceBar: {
    height: '100%',
    borderRadius: 4,
  },
  legend: {
    marginTop: 12,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 4,
  },
  legendDot: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  legendText: {
    fontSize: 13,
    color: '#666',
  },
});








