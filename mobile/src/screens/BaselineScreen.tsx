import React, { useState } from 'react';
import { View, StyleSheet, ScrollView, Alert } from 'react-native';
import { Card, Title, Paragraph, TextInput, Button } from 'react-native-paper';
import { setBaseline } from '../services/api';

export default function BaselineScreen({ route, navigation }: any) {
  const analysis = route.params?.analysis;
  
  const [actualCalories, setActualCalories] = useState('');
  const [notes, setNotes] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  const handleSaveBaseline = async () => {
    if (!analysis) {
      Alert.alert('Error', 'No analysis data available');
      return;
    }

    setIsSaving(true);
    try {
      await setBaseline(
        'default',
        analysis.analysis_id,
        actualCalories ? parseInt(actualCalories) : undefined,
        notes || undefined
      );
      
      Alert.alert(
        'Success',
        'Baseline saved! This will help improve future estimates.',
        [
          {
            text: 'OK',
            onPress: () => navigation.navigate('Home'),
          },
        ]
      );
    } catch (error) {
      Alert.alert('Error', 'Failed to save baseline');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.title}>What is Baseline?</Title>
          <Paragraph style={styles.description}>
            Baseline calibration helps improve accuracy by comparing estimates to actual values. 
            You can optionally provide the actual calorie count if you know it.
          </Paragraph>
        </Card.Content>
      </Card>

      {analysis && (
        <Card style={styles.card}>
          <Card.Content>
            <Title style={styles.sectionTitle}>Current Analysis</Title>
            <Paragraph>
              Foods: {analysis.foods.map((f: any) => f.name).join(', ')}
            </Paragraph>
            <Paragraph style={styles.estimateText}>
              Estimated: {analysis.calories_estimate} kcal
            </Paragraph>
            <Paragraph style={styles.rangeText}>
              Range: {analysis.calories_min} - {analysis.calories_max} kcal
            </Paragraph>
          </Card.Content>
        </Card>
      )}

      <Card style={styles.card}>
        <Card.Content>
          <Title style={styles.sectionTitle}>Calibration Data (Optional)</Title>
          
          <TextInput
            label="Actual Calories (if known)"
            value={actualCalories}
            onChangeText={setActualCalories}
            keyboardType="numeric"
            mode="outlined"
            style={styles.input}
            placeholder="e.g., 650"
            left={<TextInput.Icon icon="food-apple" />}
          />

          <TextInput
            label="Notes"
            value={notes}
            onChangeText={setNotes}
            mode="outlined"
            multiline
            numberOfLines={4}
            style={styles.input}
            placeholder="e.g., Homemade pasta with chicken, medium portion"
            left={<TextInput.Icon icon="note-text" />}
          />

          <Paragraph style={styles.helpText}>
            💡 Even without actual calories, saving this as a baseline helps track similar meals over time.
          </Paragraph>

          <Button
            mode="contained"
            onPress={handleSaveBaseline}
            loading={isSaving}
            disabled={isSaving || !analysis}
            style={styles.saveButton}
            icon="content-save"
          >
            Save Baseline
          </Button>
        </Card.Content>
      </Card>

      <Card style={styles.infoCard}>
        <Card.Content>
          <Title style={styles.sectionTitle}>How to Use Baseline</Title>
          <Paragraph style={styles.step}>1. Analyze a meal normally</Paragraph>
          <Paragraph style={styles.step}>2. Save it as a baseline reference</Paragraph>
          <Paragraph style={styles.step}>3. Future similar meals will be compared</Paragraph>
          <Paragraph style={styles.step}>4. Track consistency and patterns over time</Paragraph>
        </Card.Content>
      </Card>

      <View style={styles.footer}>
        <Button
          mode="outlined"
          onPress={() => navigation.navigate('Home')}
          style={styles.button}
        >
          Back to Home
        </Button>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  card: {
    margin: 16,
    elevation: 2,
  },
  infoCard: {
    margin: 16,
    backgroundColor: '#E3F2FD',
  },
  title: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#4CAF50',
    marginBottom: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
    color: '#4CAF50',
  },
  description: {
    fontSize: 15,
    lineHeight: 22,
  },
  estimateText: {
    fontSize: 18,
    fontWeight: 'bold',
    marginTop: 8,
    color: '#4CAF50',
  },
  rangeText: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  input: {
    marginBottom: 16,
  },
  helpText: {
    fontSize: 13,
    fontStyle: 'italic',
    color: '#666',
    marginBottom: 16,
  },
  saveButton: {
    marginTop: 8,
    borderRadius: 8,
  },
  step: {
    fontSize: 15,
    marginVertical: 4,
    lineHeight: 22,
  },
  footer: {
    padding: 16,
  },
  button: {
    borderRadius: 8,
  },
});








