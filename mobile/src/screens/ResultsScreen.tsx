import React, { useState, useEffect } from 'react';
import { View, StyleSheet, ScrollView, Image, Dimensions } from 'react-native';
import { Card, Title, Paragraph, Button, Chip, Divider } from 'react-native-paper';
import { AnalysisResponse } from '../services/api';
import CalorieGauge from '../components/CalorieGauge';
import { compareToBaseline } from '../services/api';

const { width } = Dimensions.get('window');

export default function ResultsScreen({ route, navigation }: any) {
  const { analysis, imageUri } = route.params as { 
    analysis: AnalysisResponse; 
    imageUri: string;
  };
  
  const [baselineComparison, setBaselineComparison] = useState<any>(null);

  useEffect(() => {
    loadBaselineComparison();
  }, []);

  const loadBaselineComparison = async () => {
    try {
      const comparison = await compareToBaseline('default', analysis.analysis_id);
      setBaselineComparison(comparison);
    } catch (error) {
      console.log('No baseline comparison available');
    }
  };

  const confidenceColor = (confidence: number) => {
    if (confidence > 0.7) return '#4CAF50';
    if (confidence > 0.4) return '#FF9800';
    return '#F44336';
  };

  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Title style={styles.headerTitle}>Analysis Results</Title>
        <Paragraph style={styles.headerSubtitle}>
          Analysis ID: {analysis.analysis_id.substring(0, 8)}
        </Paragraph>
      </View>

      <Card style={styles.imageCard}>
        <Card.Cover source={{ uri: imageUri }} />
      </Card>

      <Card style={styles.primaryCard}>
        <Card.Content>
          <View style={styles.cardHeader}>
            <Title style={styles.sectionTitle}>Energy Estimate</Title>
            <Chip
              icon="check-circle"
              style={[styles.confidenceBadge, { backgroundColor: confidenceColor(analysis.confidence) }]}
              textStyle={styles.confidenceBadgeText}
            >
              {Math.round(analysis.confidence * 100)}%
            </Chip>
          </View>
          
          <View style={styles.calorieMain}>
            <View style={styles.calorieValue}>
              <Paragraph style={styles.calorieLabel}>Estimated</Paragraph>
              <Title style={styles.calorieNumber}>{analysis.calories_estimate}</Title>
              <Paragraph style={styles.calorieUnit}>kcal</Paragraph>
            </View>
          </View>

          <View style={styles.rangeBar}>
            <View style={styles.rangeInfo}>
              <Paragraph style={styles.rangeLabel}>Range</Paragraph>
              <Paragraph style={styles.rangeValues}>
                {analysis.calories_min} - {analysis.calories_max} kcal
              </Paragraph>
            </View>
            <View style={styles.uncertaintyInfo}>
              <Paragraph style={styles.uncertaintyLabel}>Uncertainty</Paragraph>
              <Paragraph style={styles.uncertaintyValue}>
                ±{Math.round((analysis.calories_max - analysis.calories_min) / 2)} kcal
              </Paragraph>
            </View>
          </View>
        </Card.Content>
      </Card>

      <Card style={styles.dataCard}>
        <Card.Content>
          <Title style={styles.sectionTitle}>Food Classification</Title>
          {analysis.foods.map((food, index) => (
            <View key={index} style={styles.foodItem}>
              <View style={styles.foodRow}>
                <View style={styles.foodInfo}>
                  <Paragraph style={styles.foodName}>{food.name}</Paragraph>
                  {analysis.portions[index] && (
                    <Paragraph style={styles.portionText}>
                      {analysis.portions[index].portion_size} • {analysis.portions[index].grams}g
                    </Paragraph>
                  )}
                </View>
                <View style={styles.confidenceContainer}>
                  <Paragraph style={styles.confidenceValue}>
                    {(food.confidence * 100).toFixed(1)}%
                  </Paragraph>
                  <View style={styles.confidenceBar}>
                    <View 
                      style={[
                        styles.confidenceBarFill, 
                        { 
                          width: `${food.confidence * 100}%`,
                          backgroundColor: confidenceColor(food.confidence)
                        }
                      ]} 
                    />
                  </View>
                </View>
              </View>
              {index < analysis.foods.length - 1 && <Divider style={styles.divider} />}
            </View>
          ))}
        </Card.Content>
      </Card>

      <Card style={styles.dataCard}>
        <Card.Content>
          <Title style={styles.sectionTitle}>Macronutrient Distribution</Title>
          <View style={styles.nutrientGrid}>
            <View style={styles.nutrientBox}>
              <View style={[styles.nutrientIndicator, { backgroundColor: '#2196F3' }]} />
              <Paragraph style={styles.nutrientLabel}>Protein</Paragraph>
              <Title style={styles.nutrientValue}>{analysis.nutrients.protein}</Title>
              <Paragraph style={styles.nutrientUnit}>grams</Paragraph>
            </View>
            <View style={styles.nutrientBox}>
              <View style={[styles.nutrientIndicator, { backgroundColor: '#4CAF50' }]} />
              <Paragraph style={styles.nutrientLabel}>Carbohydrates</Paragraph>
              <Title style={styles.nutrientValue}>{analysis.nutrients.carbs}</Title>
              <Paragraph style={styles.nutrientUnit}>grams</Paragraph>
            </View>
            <View style={styles.nutrientBox}>
              <View style={[styles.nutrientIndicator, { backgroundColor: '#FF9800' }]} />
              <Paragraph style={styles.nutrientLabel}>Fat</Paragraph>
              <Title style={styles.nutrientValue}>{analysis.nutrients.fat}</Title>
              <Paragraph style={styles.nutrientUnit}>grams</Paragraph>
            </View>
          </View>
        </Card.Content>
      </Card>

      {baselineComparison && baselineComparison.has_baseline && (
        <Card style={styles.baselineCard}>
          <Card.Content>
            <Title style={styles.sectionTitle}>Baseline Deviation</Title>
            {baselineComparison.similar_meals > 0 ? (
              <View style={styles.baselineContent}>
                <Paragraph style={styles.baselineLabel}>
                  Compared to {baselineComparison.similar_meals} similar meal(s)
                </Paragraph>
                <View style={styles.deviationBox}>
                  <Title style={[
                    styles.deviationValue,
                    { color: baselineComparison.difference > 0 ? '#F44336' : '#4CAF50' }
                  ]}>
                    {baselineComparison.difference > 0 ? '+' : ''}
                    {baselineComparison.difference}
                  </Title>
                  <Paragraph style={styles.deviationUnit}>kcal</Paragraph>
                  <Paragraph style={styles.deviationPercent}>
                    ({baselineComparison.difference_percent > 0 ? '+' : ''}
                    {baselineComparison.difference_percent}%)
                  </Paragraph>
                </View>
              </View>
            ) : (
              <Paragraph style={styles.noBaseline}>
                No baseline data available for comparison.
              </Paragraph>
            )}
          </Card.Content>
        </Card>
      )}

      <Card style={styles.metaCard}>
        <Card.Content>
          <Title style={styles.sectionTitle}>Model Information</Title>
          <View style={styles.metaRow}>
            <Paragraph style={styles.metaLabel}>Architecture</Paragraph>
            <Paragraph style={styles.metaValue}>EfficientNet-B4</Paragraph>
          </View>
          <Divider style={styles.metaDivider} />
          <View style={styles.metaRow}>
            <Paragraph style={styles.metaLabel}>Dataset</Paragraph>
            <Paragraph style={styles.metaValue}>Food-101</Paragraph>
          </View>
          <Divider style={styles.metaDivider} />
          <View style={styles.metaRow}>
            <Paragraph style={styles.metaLabel}>Validation Accuracy</Paragraph>
            <Paragraph style={styles.metaValue}>86.0%</Paragraph>
          </View>
        </Card.Content>
      </Card>

      <View style={styles.actionButtons}>
        <Button
          mode="contained"
          icon="bookmark-outline"
          onPress={() => navigation.navigate('Baseline', { analysis })}
          style={styles.primaryActionButton}
          labelStyle={styles.primaryActionLabel}
        >
          Save as Baseline
        </Button>
        
        <View style={styles.secondaryActions}>
          <Button
            mode="outlined"
            icon="camera"
            onPress={() => navigation.navigate('Camera')}
            style={styles.secondaryActionButton}
            labelStyle={styles.secondaryActionLabel}
          >
            New Analysis
          </Button>
          
          <Button
            mode="outlined"
            icon="home-outline"
            onPress={() => navigation.navigate('Home')}
            style={styles.secondaryActionButton}
            labelStyle={styles.secondaryActionLabel}
          >
            Home
          </Button>
        </View>
      </View>

      <Card style={styles.disclaimerCard}>
        <Card.Content>
          <Title style={styles.disclaimerTitle}>Uncertainty Notice</Title>
          <Paragraph style={styles.disclaimer}>
            Results are probabilistic estimates derived from computer vision analysis. 
            Confidence intervals reflect model uncertainty. For critical applications, 
            validate results with laboratory analysis or professional assessment.
          </Paragraph>
        </Card.Content>
      </Card>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F8F9FA',
  },
  header: {
    backgroundColor: '#0A1F44',
    padding: 20,
    paddingTop: 60,
    paddingBottom: 20,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: '700',
    color: '#FFFFFF',
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#B0BEC5',
    marginTop: 4,
  },
  imageCard: {
    margin: 16,
    marginBottom: 12,
    elevation: 2,
    borderRadius: 8,
  },
  primaryCard: {
    margin: 16,
    marginTop: 0,
    marginBottom: 12,
    elevation: 2,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
  },
  dataCard: {
    margin: 16,
    marginTop: 0,
    marginBottom: 12,
    elevation: 2,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
  },
  baselineCard: {
    margin: 16,
    marginTop: 0,
    marginBottom: 12,
    elevation: 2,
    borderRadius: 8,
    backgroundColor: '#E3F2FD',
    borderLeftWidth: 4,
    borderLeftColor: '#2196F3',
  },
  metaCard: {
    margin: 16,
    marginTop: 0,
    marginBottom: 12,
    elevation: 1,
    borderRadius: 8,
    backgroundColor: '#FAFAFA',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#0A1F44',
  },
  confidenceBadge: {
    height: 28,
  },
  confidenceBadgeText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '600',
  },
  calorieMain: {
    alignItems: 'center',
    marginVertical: 20,
    paddingVertical: 8,
  },
  calorieValue: {
    alignItems: 'center',
  },
  calorieLabel: {
    fontSize: 13,
    color: '#78909C',
    textTransform: 'uppercase',
    letterSpacing: 1,
    marginBottom: 4,
  },
  calorieNumber: {
    fontSize: 56,
    fontWeight: '700',
    color: '#0A1F44',
    marginVertical: 4,
    lineHeight: 64,
  },
  calorieUnit: {
    fontSize: 16,
    color: '#78909C',
    marginTop: 4,
  },
  rangeBar: {
    backgroundColor: '#F5F5F5',
    borderRadius: 8,
    padding: 16,
    marginTop: 12,
  },
  rangeInfo: {
    marginBottom: 8,
  },
  rangeLabel: {
    fontSize: 12,
    color: '#78909C',
    marginBottom: 4,
  },
  rangeValues: {
    fontSize: 16,
    fontWeight: '600',
    color: '#546E7A',
  },
  uncertaintyInfo: {
    marginTop: 8,
  },
  uncertaintyLabel: {
    fontSize: 12,
    color: '#78909C',
    marginBottom: 4,
  },
  uncertaintyValue: {
    fontSize: 14,
    fontWeight: '500',
    color: '#FF9800',
  },
  foodItem: {
    marginVertical: 12,
  },
  foodRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  foodInfo: {
    flex: 1,
    marginRight: 16,
  },
  foodName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#0A1F44',
    textTransform: 'capitalize',
    marginBottom: 4,
  },
  portionText: {
    fontSize: 13,
    color: '#78909C',
    textTransform: 'capitalize',
  },
  confidenceContainer: {
    alignItems: 'flex-end',
    minWidth: 80,
  },
  confidenceValue: {
    fontSize: 14,
    fontWeight: '600',
    color: '#546E7A',
    marginBottom: 4,
  },
  confidenceBar: {
    width: 80,
    height: 6,
    backgroundColor: '#E0E0E0',
    borderRadius: 3,
    overflow: 'hidden',
  },
  confidenceBarFill: {
    height: '100%',
    borderRadius: 3,
  },
  divider: {
    marginTop: 12,
  },
  nutrientGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 12,
  },
  nutrientBox: {
    flex: 1,
    alignItems: 'center',
    padding: 12,
    marginHorizontal: 4,
    backgroundColor: '#F8F9FA',
    borderRadius: 8,
  },
  nutrientIndicator: {
    width: 32,
    height: 4,
    borderRadius: 2,
    marginBottom: 8,
  },
  nutrientLabel: {
    fontSize: 11,
    color: '#78909C',
    textAlign: 'center',
    marginBottom: 8,
  },
  nutrientValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#0A1F44',
  },
  nutrientUnit: {
    fontSize: 11,
    color: '#78909C',
    marginTop: 2,
  },
  baselineContent: {
    alignItems: 'center',
  },
  baselineLabel: {
    fontSize: 14,
    color: '#546E7A',
    marginBottom: 16,
  },
  deviationBox: {
    alignItems: 'center',
  },
  deviationValue: {
    fontSize: 36,
    fontWeight: '700',
  },
  deviationUnit: {
    fontSize: 14,
    color: '#78909C',
    marginTop: 4,
  },
  deviationPercent: {
    fontSize: 16,
    color: '#78909C',
    marginTop: 2,
  },
  noBaseline: {
    fontSize: 14,
    color: '#78909C',
    textAlign: 'center',
  },
  metaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
  },
  metaLabel: {
    fontSize: 13,
    color: '#78909C',
  },
  metaValue: {
    fontSize: 13,
    fontWeight: '600',
    color: '#546E7A',
  },
  metaDivider: {
    marginVertical: 4,
  },
  actionButtons: {
    padding: 16,
    paddingTop: 8,
  },
  primaryActionButton: {
    marginBottom: 12,
    borderRadius: 6,
    backgroundColor: '#2196F3',
    elevation: 0,
  },
  primaryActionLabel: {
    fontSize: 15,
    fontWeight: '500',
    paddingVertical: 8,
  },
  secondaryActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  secondaryActionButton: {
    flex: 1,
    marginHorizontal: 4,
    borderRadius: 6,
    borderColor: '#CFD8DC',
  },
  secondaryActionLabel: {
    fontSize: 14,
    color: '#546E7A',
  },
  disclaimerCard: {
    margin: 16,
    marginTop: 0,
    marginBottom: 20,
    elevation: 1,
    borderRadius: 8,
    backgroundColor: '#FFF8E1',
    borderLeftWidth: 4,
    borderLeftColor: '#FFA000',
  },
  disclaimerTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#E65100',
    marginBottom: 8,
  },
  disclaimer: {
    fontSize: 12,
    lineHeight: 18,
    color: '#F57C00',
  },
});




