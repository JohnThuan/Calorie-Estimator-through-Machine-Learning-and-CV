import React, { useState, useRef } from 'react';
import { View, StyleSheet, TouchableOpacity, Image, Alert } from 'react-native';
import { CameraView, CameraType, useCameraPermissions } from 'expo-camera';
import { Button, ActivityIndicator, Text } from 'react-native-paper';
import * as ImagePicker from 'expo-image-picker';
import { analyzeImage } from '../services/api';

export default function CameraScreen({ navigation }: any) {
  const [permission, requestPermission] = useCameraPermissions();
  const [facing, setFacing] = useState<CameraType>('back');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const cameraRef = useRef<CameraView>(null);

  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 0.8,
        });
        if (photo) {
          setCapturedImage(photo.uri);
        }
      } catch (error) {
        console.error('Error taking picture:', error);
        Alert.alert('Error', 'Failed to capture image');
      }
    }
  };

  const pickImage = async () => {
    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [4, 3],
        quality: 0.8,
      });

      if (!result.canceled) {
        setCapturedImage(result.assets[0].uri);
      }
    } catch (error) {
      console.error('Error picking image:', error);
      Alert.alert('Error', 'Failed to select image');
    }
  };

  const analyzePhoto = async () => {
    if (!capturedImage) return;

    setIsAnalyzing(true);
    try {
      const result = await analyzeImage(capturedImage);
      navigation.navigate('Results', { analysis: result, imageUri: capturedImage });
    } catch (error) {
      console.error('Analysis error:', error);
      Alert.alert(
        'Analysis Failed',
        'Could not analyze the image. Please make sure the backend server is running.',
        [{ text: 'OK' }]
      );
    } finally {
      setIsAnalyzing(false);
    }
  };

  const retake = () => {
    setCapturedImage(null);
  };

  if (!permission) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.permissionText}>Camera permission is required</Text>
        <Button mode="contained" onPress={requestPermission} style={styles.button}>
          Grant Permission
        </Button>
        <Button mode="outlined" onPress={pickImage} style={styles.button}>
          Select from Gallery Instead
        </Button>
      </View>
    );
  }

  if (capturedImage) {
    return (
      <View style={styles.container}>
        <Image source={{ uri: capturedImage }} style={styles.preview} />
        
        {isAnalyzing ? (
          <View style={styles.analyzingContainer}>
            <ActivityIndicator size="large" color="#2196F3" />
            <Text style={styles.analyzingTitle}>Processing Image</Text>
            <Text style={styles.analyzingSubtitle}>Applying Vision Transformer model...</Text>
          </View>
        ) : (
          <View style={styles.buttonContainer}>
            <Text style={styles.reviewTitle}>Review Captured Image</Text>
            <Text style={styles.reviewSubtitle}>
              Ensure image is clear and food is well-lit
            </Text>
            <Button
              mode="contained"
              onPress={analyzePhoto}
              style={styles.analyzeButton}
              contentStyle={styles.buttonContent}
              labelStyle={styles.analyzeButtonLabel}
              icon="arrow-right"
            >
              Run Analysis
            </Button>
            <Button
              mode="outlined"
              onPress={retake}
              style={styles.retakeButton}
              contentStyle={styles.buttonContent}
              labelStyle={styles.retakeButtonLabel}
              icon="camera-retake"
            >
              Retake Photo
            </Button>
          </View>
        )}
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} facing={facing} ref={cameraRef} />
      <View style={styles.cameraOverlay}>
          <View style={styles.topBar}>
            <TouchableOpacity 
              onPress={() => navigation.goBack()}
              style={styles.backButton}
            >
              <Text style={styles.backButtonText}>← Cancel</Text>
            </TouchableOpacity>
            <Text style={styles.topBarTitle}>Image Acquisition</Text>
            <View style={styles.placeholder} />
          </View>

          <View style={styles.frameContainer}>
            <View style={styles.frameCornerTL} />
            <View style={styles.frameCornerTR} />
            <View style={styles.frameCornerBL} />
            <View style={styles.frameCornerBR} />
            <View style={styles.instructionBox}>
              <Text style={styles.instructionText}>
                Center food item within frame
              </Text>
              <Text style={styles.instructionSubtext}>
                Ensure adequate lighting and focus
              </Text>
            </View>
          </View>

          <View style={styles.bottomPanel}>
            <TouchableOpacity
              onPress={pickImage}
              style={styles.sideControl}
            >
              <Text style={styles.controlIcon}>📁</Text>
              <Text style={styles.controlLabel}>Gallery</Text>
            </TouchableOpacity>
            
            <TouchableOpacity onPress={takePicture} style={styles.captureButton}>
              <View style={styles.captureButtonOuter}>
                <View style={styles.captureButtonInner} />
              </View>
            </TouchableOpacity>
            
            <TouchableOpacity
              onPress={() => {
                setFacing(current => (current === 'back' ? 'front' : 'back'));
              }}
              style={styles.sideControl}
            >
              <Text style={styles.controlIcon}>🔄</Text>
              <Text style={styles.controlLabel}>Flip</Text>
            </TouchableOpacity>
          </View>
        </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
    backgroundColor: '#F8F9FA',
  },
  camera: {
    flex: 1,
  },
  cameraOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'transparent',
    justifyContent: 'space-between',
  },
  topBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingTop: 50,
    paddingHorizontal: 20,
    paddingBottom: 20,
    backgroundColor: 'rgba(10, 31, 68, 0.9)',
  },
  backButton: {
    padding: 8,
  },
  backButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '500',
  },
  topBarTitle: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  placeholder: {
    width: 70,
  },
  frameContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  frameCornerTL: {
    position: 'absolute',
    top: '25%',
    left: '15%',
    width: 40,
    height: 40,
    borderTopWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#2196F3',
  },
  frameCornerTR: {
    position: 'absolute',
    top: '25%',
    right: '15%',
    width: 40,
    height: 40,
    borderTopWidth: 3,
    borderRightWidth: 3,
    borderColor: '#2196F3',
  },
  frameCornerBL: {
    position: 'absolute',
    bottom: '25%',
    left: '15%',
    width: 40,
    height: 40,
    borderBottomWidth: 3,
    borderLeftWidth: 3,
    borderColor: '#2196F3',
  },
  frameCornerBR: {
    position: 'absolute',
    bottom: '25%',
    right: '15%',
    width: 40,
    height: 40,
    borderBottomWidth: 3,
    borderRightWidth: 3,
    borderColor: '#2196F3',
  },
  instructionBox: {
    position: 'absolute',
    bottom: '20%',
    backgroundColor: 'rgba(10, 31, 68, 0.85)',
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderRadius: 8,
  },
  instructionText: {
    color: '#FFFFFF',
    fontSize: 15,
    fontWeight: '500',
    textAlign: 'center',
  },
  instructionSubtext: {
    color: '#B0BEC5',
    fontSize: 12,
    textAlign: 'center',
    marginTop: 4,
  },
  bottomPanel: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    paddingBottom: 50,
    paddingTop: 20,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(10, 31, 68, 0.9)',
  },
  sideControl: {
    alignItems: 'center',
    padding: 10,
  },
  controlIcon: {
    fontSize: 24,
  },
  controlLabel: {
    color: '#FFFFFF',
    fontSize: 12,
    marginTop: 6,
    fontWeight: '500',
  },
  captureButton: {
    padding: 4,
  },
  captureButtonOuter: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: 'rgba(255,255,255,0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 3,
    borderColor: '#FFFFFF',
  },
  captureButtonInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: '#2196F3',
  },
  preview: {
    flex: 1,
    resizeMode: 'contain',
    backgroundColor: '#000',
  },
  buttonContainer: {
    padding: 24,
    backgroundColor: '#FFFFFF',
  },
  reviewTitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#0A1F44',
    marginBottom: 8,
  },
  reviewSubtitle: {
    fontSize: 14,
    color: '#78909C',
    marginBottom: 24,
  },
  buttonContent: {
    height: 52,
  },
  analyzeButton: {
    backgroundColor: '#2196F3',
    marginBottom: 12,
    borderRadius: 6,
    elevation: 0,
  },
  analyzeButtonLabel: {
    fontSize: 16,
    fontWeight: '600',
    letterSpacing: 0.5,
  },
  retakeButton: {
    borderColor: '#CFD8DC',
    borderWidth: 1,
    borderRadius: 6,
  },
  retakeButtonLabel: {
    fontSize: 16,
    fontWeight: '500',
    color: '#546E7A',
  },
  analyzingContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(10, 31, 68, 0.95)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  analyzingTitle: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: '600',
    marginTop: 24,
  },
  analyzingSubtitle: {
    color: '#B0BEC5',
    fontSize: 14,
    marginTop: 8,
  },
  permissionText: {
    fontSize: 16,
    color: '#546E7A',
    marginBottom: 24,
    textAlign: 'center',
    lineHeight: 24,
  },
  button: {
    marginVertical: 8,
    borderRadius: 6,
  },
});


