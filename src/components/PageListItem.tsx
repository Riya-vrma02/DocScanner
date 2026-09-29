import React from 'react';
import {View,Text,Image,TouchableOpacity,StyleSheet} from 'react-native';
import {ScanPage} from '../types';

interface Props{
  page:ScanPage;
  index:number;
  onDelete:()=>void;
  onEdit:()=>void;
  onLongPress?:()=>void;
}

export default function PageListItem({page,index,onDelete,onEdit,onLongPress}:Props){
  return(
    <TouchableOpacity style={styles.card} onPress={onEdit} onLongPress={onLongPress} delayLongPress={150}>
      <Image source={{uri:`file://${page.processedImagePath}`}} style={styles.thumb} />
      <Text style={styles.label}>Page{index+1}</Text>
      <TouchableOpacity onPress={onDelete} style={styles.deleteBtn}>
        <Text style={styles.deleteText}>✕</Text>
      </TouchableOpacity>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row', alignItems: 'center', backgroundColor: '#fff',
    borderRadius: 8, padding: 8, marginVertical: 4, marginHorizontal: 8, elevation: 2,
  },
  thumb: { width: 56, height: 76, borderRadius: 4, backgroundColor: '#eee' },
  // Explicit color: without it, the label inherits a light default in system
  // dark mode and disappears against the white card.
  label: { flex: 1, marginLeft: 12, fontSize: 16, color: '#111' },
  deleteBtn: { padding: 8 },
  deleteText: { color: '#D32F2F', fontSize: 16 },
});
