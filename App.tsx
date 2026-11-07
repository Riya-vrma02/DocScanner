import 'react-native-gesture-handler';
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import HomeScreen from './src/screens/HomeScreen';
import CameraScanScreen from './src/screens/CameraScanScreen';
import CropScreen from './src/screens/CropScreen';
import IrregularCropScreen from './src/screens/IrregularCropScreen';
import MultiDocPickerScreen from './src/screens/MultiDocPickerScreen';
import FilterScreen from './src/screens/FilterScreen';
import ReviewScreen from './src/screens/ReviewScreen';

export type RootStackParamList = {
  Home: undefined;
  /** Full-size page view, opened on the tapped page. */
  Review: { initialIndex?: number } | undefined;
  /** replacePageId: retake that page instead of adding a new one. */
  Scan: { replacePageId?: string } | undefined;
  /** replacePageId: overwrite that page (retake / re-crop) instead of adding a new one. */
  Crop: {
    rawPath: string; corners: { x: number; y: number }[]; imageWidth: number; imageHeight: number;
    replacePageId?: string;
  };
  IrregularCrop: { rawPath: string; points: { x: number; y: number }[] };
  MultiDocPicker: { rawPath: string; documents: any[]; imageWidth: number; imageHeight: number };
  Filter: { pageId: string };
};

const Stack = createNativeStackNavigator<RootStackParamList>();

export default function App() {
  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <NavigationContainer>
        <Stack.Navigator>
          <Stack.Screen
            name="Home"
            component={HomeScreen}
            options={{
              title: 'DocScanner',
              headerStyle: { backgroundColor: '#1F2125' },
              headerTintColor: '#FFFFFF',
              headerShadowVisible: false,
            }}
          />
          <Stack.Screen
            name="Review"
            component={ReviewScreen}
            options={{
              title: 'Pages',
              headerStyle: { backgroundColor: '#1F2125' },
              headerTintColor: '#FFFFFF',
              headerShadowVisible: false,
            }}
          />
          <Stack.Screen name="Scan" component={CameraScanScreen} options={{ headerShown: false }} />
          <Stack.Screen name="Crop" component={CropScreen} options={{ title: 'Adjust corners' }} />
          <Stack.Screen name="IrregularCrop" component={IrregularCropScreen} options={{ title: 'Confirm crop' }} />
          <Stack.Screen name="MultiDocPicker" component={MultiDocPickerScreen} options={{ title: 'Select pages' }} />
          <Stack.Screen name="Filter" component={FilterScreen} options={{ title: 'Edit page' }} />
        </Stack.Navigator>
      </NavigationContainer>
    </GestureHandlerRootView>
  );
}