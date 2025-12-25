import React from 'react';
import { View, StyleSheet, ScrollView, Text } from 'react-native';
import { Button, Card, Title, Paragraph, Divider } from 'react-native-paper';

export default function HomeScreen({ navigation }: any) {
  return (
    <ScrollView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.logo}>CALORIE ESTIMATION</Text>
        <Text style={styles.subtitle}>Through Computer Vision</Text>
        <Text style={styles.version}>Research Project v1.0</Text>
      </View>

      <View style={styles.content}>
        <Card style={styles.heroCard}>
          <Card.Content>
            <Title style={styles.heroTitle}>Computer Vision for Dietary Assessment</Title>
            <Paragraph style={styles.heroDescription}>
              Deep learning-based food recognition and calorie estimation using EfficientNet-B4 
              architecture. Trained on Food-101 dataset with 86% validation accuracy.
            </Paragraph>
            <Divider style={styles.divider} />
            <View style={styles.statsRow}>
              <View style={styles.stat}>
                <Text style={styles.statValue}>101</Text>
                <Text style={styles.statLabel}>Food Classes</Text>
              </View>
              <View style={styles.stat}>
                <Text style={styles.statValue}>86%</Text>
                <Text style={styles.statLabel}>Accuracy</Text>
              </View>
              <View style={styles.stat}>
                <Text style={styles.statValue}>19M</Text>
                <Text style={styles.statLabel}>Parameters</Text>
              </View>
            </View>
          </Card.Content>
        </Card>

        <Card style={styles.actionCard}>
          <Card.Content>
            <Title style={styles.cardTitle}>Analysis Tools</Title>
            <Button
              mode="contained"
              icon="camera"
              style={styles.primaryButton}
              onPress={() => navigation.navigate('Camera')}
              contentStyle={styles.buttonContent}
              labelStyle={styles.buttonLabel}
            >
              New Analysis
            </Button>

            <Button
              mode="outlined"
              icon="history"
              style={styles.secondaryButton}
              onPress={() => navigation.navigate('History')}
              contentStyle={styles.buttonContent}
              labelStyle={styles.secondaryButtonLabel}
            >
              Analysis History
            </Button>

            <Button
              mode="outlined"
              icon="chart-box-outline"
              style={styles.secondaryButton}
              onPress={() => navigation.navigate('Baseline')}
              contentStyle={styles.buttonContent}
              labelStyle={styles.secondaryButtonLabel}
            >
              Baseline Comparison
            </Button>
          </Card.Content>
        </Card>

        <Card style={styles.methodCard}>
          <Card.Content>
            <Title style={styles.cardTitle}>Methodology</Title>
            <View style={styles.methodStep}>
              <Text style={styles.stepNumber}>1</Text>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>Image Acquisition</Text>
                <Text style={styles.stepDesc}>High-resolution RGB image capture</Text>
              </View>
            </View>
            <View style={styles.methodStep}>
              <Text style={styles.stepNumber}>2</Text>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>Feature Extraction</Text>
                <Text style={styles.stepDesc}>EfficientNet-B4 deep convolutional processing</Text>
              </View>
            </View>
            <View style={styles.methodStep}>
              <Text style={styles.stepNumber}>3</Text>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>Classification</Text>
                <Text style={styles.stepDesc}>Multi-class food identification with confidence scores</Text>
              </View>
            </View>
            <View style={styles.methodStep}>
              <Text style={styles.stepNumber}>4</Text>
              <View style={styles.stepContent}>
                <Text style={styles.stepTitle}>Nutritional Estimation</Text>
                <Text style={styles.stepDesc}>Portion-based calorie and macronutrient calculation</Text>
              </View>
            </View>
          </Card.Content>
        </Card>

        <Card style={styles.disclaimerCard}>
          <Card.Content>
            <Title style={styles.disclaimerTitle}>Research Notice</Title>
            <Paragraph style={styles.disclaimer}>
              This system is a research prototype for educational and demonstration purposes. 
              Results are probabilistic estimates and should not be used for medical, clinical, 
              or professional dietary planning without validation by qualified professionals.
            </Paragraph>
          </Card.Content>
        </Card>
      </View>
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
    paddingTop: 60,
    paddingBottom: 30,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  logo: {
    fontSize: 32,
    fontWeight: '700',
    color: '#FFFFFF',
    letterSpacing: 4,
  },
  subtitle: {
    fontSize: 14,
    color: '#B0BEC5',
    marginTop: 8,
    letterSpacing: 1,
  },
  version: {
    fontSize: 11,
    color: '#78909C',
    marginTop: 4,
  },
  content: {
    padding: 16,
  },
  heroCard: {
    marginBottom: 16,
    elevation: 2,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
  },
  heroTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#0A1F44',
    marginBottom: 12,
  },
  heroDescription: {
    fontSize: 14,
    lineHeight: 22,
    color: '#546E7A',
  },
  divider: {
    marginVertical: 16,
  },
  statsRow: {
    flexDirection: 'row',
    justifyContent: 'space-around',
  },
  stat: {
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: '700',
    color: '#2196F3',
  },
  statLabel: {
    fontSize: 12,
    color: '#78909C',
    marginTop: 4,
  },
  actionCard: {
    marginBottom: 16,
    elevation: 2,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
  },
  cardTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#0A1F44',
    marginBottom: 16,
  },
  primaryButton: {
    marginBottom: 12,
    borderRadius: 6,
    backgroundColor: '#2196F3',
    elevation: 0,
  },
  secondaryButton: {
    marginBottom: 12,
    borderRadius: 6,
    borderColor: '#CFD8DC',
    borderWidth: 1,
  },
  buttonContent: {
    height: 50,
  },
  buttonLabel: {
    fontSize: 15,
    fontWeight: '500',
    letterSpacing: 0.5,
  },
  secondaryButtonLabel: {
    fontSize: 15,
    fontWeight: '500',
    color: '#546E7A',
    letterSpacing: 0.5,
  },
  methodCard: {
    marginBottom: 16,
    elevation: 2,
    borderRadius: 8,
    backgroundColor: '#FFFFFF',
  },
  methodStep: {
    flexDirection: 'row',
    marginBottom: 16,
    alignItems: 'flex-start',
  },
  stepNumber: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#E3F2FD',
    color: '#2196F3',
    fontSize: 16,
    fontWeight: '600',
    textAlign: 'center',
    lineHeight: 32,
    marginRight: 12,
  },
  stepContent: {
    flex: 1,
  },
  stepTitle: {
    fontSize: 15,
    fontWeight: '600',
    color: '#0A1F44',
    marginBottom: 4,
  },
  stepDesc: {
    fontSize: 13,
    color: '#78909C',
    lineHeight: 18,
  },
  disclaimerCard: {
    marginBottom: 20,
    elevation: 1,
    borderRadius: 8,
    backgroundColor: '#FFF8E1',
    borderLeftWidth: 4,
    borderLeftColor: '#FFA000',
  },
  disclaimerTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#E65100',
    marginBottom: 8,
  },
  disclaimer: {
    fontSize: 13,
    lineHeight: 20,
    color: '#F57C00',
  },
});




