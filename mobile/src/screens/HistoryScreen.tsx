import React, { useEffect, useState } from 'react';
import { View, StyleSheet, ScrollView, Alert } from 'react-native';
import { Card, Title, Paragraph, Button, ActivityIndicator, Chip } from 'react-native-paper';
import { getHistory } from '../services/api';

export default function HistoryScreen({ navigation }: any) {
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = async () => {
    try {
      const data = await getHistory();
      setHistory(data.entries);
    } catch (error) {
      Alert.alert('Error', 'Failed to load history');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (loading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#4CAF50" />
      </View>
    );
  }

  if (history.length === 0) {
    return (
      <View style={styles.centerContainer}>
        <Title>No History Yet</Title>
        <Paragraph style={styles.emptyText}>
          Start analyzing meals to build your history!
        </Paragraph>
        <Button
          mode="contained"
          icon="camera"
          onPress={() => navigation.navigate('Camera')}
          style={styles.button}
        >
          Analyze First Meal
        </Button>
      </View>
    );
  }

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Title>Your Meal History</Title>
        <Paragraph>{history.length} meal(s) analyzed</Paragraph>
      </View>

      {history.map((entry, index) => (
        <Card key={entry.id} style={styles.card}>
          <Card.Content>
            <View style={styles.cardHeader}>
              <Paragraph style={styles.date}>{formatDate(entry.timestamp)}</Paragraph>
              <Chip style={styles.calorieChip}>
                {entry.calories.estimate} kcal
              </Chip>
            </View>
            
            <View style={styles.foodsContainer}>
              {entry.foods.map((food: any, i: number) => (
                <Chip key={i} style={styles.foodChip} textStyle={styles.foodChipText}>
                  {food.name}
                </Chip>
              ))}
            </View>

            <View style={styles.details}>
              <Paragraph style={styles.detailText}>
                Range: {entry.calories.min} - {entry.calories.max} kcal
              </Paragraph>
              <Paragraph style={styles.detailText}>
                Confidence: {Math.round(entry.calories.confidence * 100)}%
              </Paragraph>
            </View>
          </Card.Content>
        </Card>
      ))}

      <View style={styles.footer}>
        <Button
          mode="outlined"
          icon="download"
          onPress={() => Alert.alert('Export', 'Export functionality coming soon!')}
          style={styles.button}
        >
          Export History
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
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  header: {
    padding: 16,
    backgroundColor: '#fff',
    marginBottom: 8,
  },
  emptyText: {
    textAlign: 'center',
    marginVertical: 20,
    fontSize: 16,
  },
  card: {
    margin: 8,
    marginHorizontal: 16,
    elevation: 2,
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  date: {
    fontSize: 14,
    color: '#666',
  },
  calorieChip: {
    backgroundColor: '#4CAF50',
  },
  foodsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginVertical: 8,
  },
  foodChip: {
    marginRight: 6,
    marginBottom: 6,
    backgroundColor: '#E8F5E9',
  },
  foodChipText: {
    textTransform: 'capitalize',
  },
  details: {
    marginTop: 8,
    paddingTop: 8,
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
  },
  detailText: {
    fontSize: 13,
    color: '#666',
  },
  footer: {
    padding: 16,
  },
  button: {
    marginVertical: 8,
    borderRadius: 8,
  },
});








