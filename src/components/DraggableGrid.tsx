import React, { useCallback, useEffect, useState } from 'react';
import { View, StyleSheet, LayoutChangeEvent } from 'react-native';
import Animated, {
  SharedValue, runOnJS, useAnimatedReaction, useAnimatedStyle, useSharedValue, withSpring, withTiming,
} from 'react-native-reanimated';
import { Gesture, GestureDetector, ScrollView } from 'react-native-gesture-handler';

/** Slot index of every item, by key. Shared with the UI thread so tiles can move while dragging. */
type Positions = { [key: string]: number };

interface Props<T> {
  data: T[];
  keyExtractor: (item: T) => string;
  /** displayIndex is the tile's CURRENT slot (it updates live while another tile is dragged). */
  renderItem: (item: T, displayIndex: number, isDragging: boolean) => React.ReactNode;
  onReorder: (newOrder: T[]) => void;
  onPress?: (item: T, index: number) => void;
  numColumns?: number;
  gap?: number;
  /** Tile height / tile width. */
  aspect?: number;
  /** Extra space below the last row (e.g. so it can scroll above a bottom bar). */
  bottomInset?: number;
}

const SPRING = { damping: 20, stiffness: 220, mass: 0.6 };

/**
 * A grid whose tiles can be long-pressed and dragged to a new place. The other tiles slide to make
 * room as the dragged tile passes over them; the new order is reported once the tile is released.
 * Built on react-native-gesture-handler + reanimated (already used by the app), no extra library.
 */
export default function DraggableGrid<T>({
  data, keyExtractor, renderItem, onReorder, onPress,
  numColumns = 3, gap = 12, aspect = 1.38, bottomInset = 0,
}: Props<T>) {
  const [width, setWidth] = useState(0);
  const [scrollEnabled, setScrollEnabled] = useState(true);
  const positions = useSharedValue<Positions>({});

  const keys = data.map(keyExtractor);
  const keysSig = keys.join('|');
  useEffect(() => {
    const p: Positions = {};
    keys.forEach((k, i) => { p[k] = i; });
    positions.value = p;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [keysSig]);

  const itemW = width > 0 ? (width - gap * (numColumns + 1)) / numColumns : 0;
  const itemH = itemW * aspect;
  const rows = Math.ceil(data.length / numColumns);
  const contentH = gap + rows * (itemH + gap) + bottomInset;

  const commit = useCallback((p: Positions) => {
    const sorted = [...data].sort((a, b) => p[keyExtractor(a)] - p[keyExtractor(b)]);
    const changed = sorted.some((item, i) => item !== data[i]);
    if (changed) onReorder(sorted);
  }, [data, keyExtractor, onReorder]);

  return (
    <ScrollView
      style={styles.flex}
      scrollEnabled={scrollEnabled}
      onLayout={(e: LayoutChangeEvent) => setWidth(e.nativeEvent.layout.width)}
    >
      <View style={{ height: contentH }}>
        {itemW > 0 && data.map((item, index) => {
          const key = keyExtractor(item);
          return (
            <GridTile
              key={key}
              id={key}
              index={index}
              count={data.length}
              positions={positions}
              itemW={itemW}
              itemH={itemH}
              gap={gap}
              cols={numColumns}
              onDragState={(d) => setScrollEnabled(!d)}
              commit={commit}
              onPress={() => onPress?.(item, positions.value[key] ?? index)}
              render={(num, dragging) => renderItem(item, num, dragging)}
            />
          );
        })}
      </View>
    </ScrollView>
  );
}

interface TileProps {
  id: string;
  index: number;
  count: number;
  positions: SharedValue<Positions>;
  itemW: number;
  itemH: number;
  gap: number;
  cols: number;
  onDragState: (dragging: boolean) => void;
  commit: (p: Positions) => void;
  onPress: () => void;
  render: (displayIndex: number, dragging: boolean) => React.ReactNode;
}

function GridTile({
  id, index, count, positions, itemW, itemH, gap, cols, onDragState, commit, onPress, render,
}: TileProps) {
  const slotX = (i: number) => {
    'worklet';
    return gap + (i % cols) * (itemW + gap);
  };
  const slotY = (i: number) => {
    'worklet';
    return gap + Math.floor(i / cols) * (itemH + gap);
  };

  const tx = useSharedValue(slotX(index));
  const ty = useSharedValue(slotY(index));
  const startX = useSharedValue(0);
  const startY = useSharedValue(0);
  const dragging = useSharedValue(false);
  const [displayIndex, setDisplayIndex] = useState(index);
  const [isDragging, setIsDragging] = useState(false);

  // Follow this tile's slot: slide there whenever another tile pushes it, and keep the number in sync.
  useAnimatedReaction(
    () => positions.value[id],
    (cur, prev) => {
      if (cur === undefined) return;
      if (cur !== prev) runOnJS(setDisplayIndex)(cur);
      if (!dragging.value) {
        tx.value = withSpring(slotX(cur), SPRING);
        ty.value = withSpring(slotY(cur), SPRING);
      }
    },
    [itemW, itemH, cols, gap]
  );

  const pan = Gesture.Pan()
    .activateAfterLongPress(220)
    .onStart(() => {
      dragging.value = true;
      startX.value = tx.value;
      startY.value = ty.value;
      runOnJS(onDragState)(true);
      runOnJS(setIsDragging)(true);
    })
    .onUpdate((e) => {
      tx.value = startX.value + e.translationX;
      ty.value = startY.value + e.translationY;
      // Which slot is the tile's centre over now?
      const col = Math.max(0, Math.min(cols - 1, Math.round((tx.value - gap) / (itemW + gap))));
      const row = Math.max(0, Math.round((ty.value - gap) / (itemH + gap)));
      const to = Math.min(count - 1, row * cols + col);
      const from = positions.value[id];
      if (to !== from) {
        // Move this tile to `to` and shift the tiles in between by one.
        const next: Positions = {};
        for (const k in positions.value) {
          const v = positions.value[k];
          if (k === id) next[k] = to;
          else if (from < to && v > from && v <= to) next[k] = v - 1;
          else if (from > to && v >= to && v < from) next[k] = v + 1;
          else next[k] = v;
        }
        positions.value = next;
      }
    })
    .onFinalize(() => {
      if (!dragging.value) return;
      dragging.value = false;
      const cur = positions.value[id];
      tx.value = withSpring(slotX(cur), SPRING);
      ty.value = withSpring(slotY(cur), SPRING);
      runOnJS(onDragState)(false);
      runOnJS(setIsDragging)(false);
      runOnJS(commit)(positions.value);
    });

  const tap = Gesture.Tap()
    .maxDuration(200)
    .onEnd((_e, success) => {
      if (success) runOnJS(onPress)();
    });

  const style = useAnimatedStyle(() => ({
    transform: [
      { translateX: tx.value },
      { translateY: ty.value },
      { scale: withTiming(dragging.value ? 1.07 : 1, { duration: 150 }) },
    ],
    zIndex: dragging.value ? 100 : 1,
    elevation: dragging.value ? 14 : 3,
    shadowOpacity: dragging.value ? 0.35 : 0.15,
  }));

  return (
    <GestureDetector gesture={Gesture.Exclusive(pan, tap)}>
      <Animated.View style={[styles.tile, { width: itemW, height: itemH }, style]}>
        {render(displayIndex, isDragging)}
      </Animated.View>
    </GestureDetector>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  tile: {
    position: 'absolute', left: 0, top: 0,
    shadowColor: '#000', shadowOffset: { width: 0, height: 3 }, shadowRadius: 6,
  },
});