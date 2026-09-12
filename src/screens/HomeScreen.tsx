import React,{useState} from 'react';
import {View,Text,StyleSheet,TouchableOpacity,Modal} from 'react-native';
import DraggableFlatList,{RenderItemParams} from 'react-native-draggable-flatlist';
import Share from 'react-native-share';
import {usePagesStore} from '../store/pagesStore';
import {exportPagesToPdf} from '../lib/pdfExport';
import {PageSizeName,ScanPage} from '../types';
import PageListItem from '../components/PageListItem';

/**shows every page captured in the session,drag to reorder, tap to delete or
 * re-edit, and "Export as PDF"*/
export default function HomeScreen({navigation}:any){
  const pages = usePagesStore((s)=>s.pages);
  const removePage = usePagesStore((s)=>s.removePage);
  const reorder = usePagesStore((s)=>s.reorder);

  const [pickerVisible,setPickerVisible] = useState(false);
  const [exporting,setExporting] = useState(false);

  async function doExport(pageSize: PageSizeName) {
    setPickerVisible(false);
    setExporting(true);
    try{
      const path = await exportPagesToPdf(pages,pageSize,`Scan_${Date.now()}.pdf`);
      await Share.open({url:`file://${path}`,type:'application/pdf'});
    }
    finally{
      setExporting(false);
    }
  }
  return (
    <View style={styles.container}>
      {pages.length===0?(
        <View style={styles.empty}>
          <Text style={styles.emptyText}>No pages yet.{'\n'}Tap "Scan page" to add your first one.</Text>
        </View>
      ):(
        <DraggableFlatList
          data={pages}
          keyExtractor={(p)=>p.id}
          onDragEnd={({data})=>reorder(data)}
          renderItem={({item,index,drag}:RenderItemParams<ScanPage>)=>(
            <PageListItem
              page={item}
              index={index??0}
              onEdit={()=>navigation.navigate('Filter',{pageId:item.id})}
              onDelete={()=>removePage(item.id)}
              onLongPress={drag}
            />
          )}
        />
      )}
      <View style={styles.bottomBar}>
        <TouchableOpacity style={styles.scanButton} onPress={()=>navigation.navigate('Scan')}>
          <Text style={styles.buttonText}>+ Scan page</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.exportButton,pages.length===0 && styles.disabled]}
          disabled={pages.length===0 || exporting}
          onPress={()=>setPickerVisible(true)}
        >
          <Text style={styles.buttonText}>{exporting?'Exporting…':'Export as PDF'}</Text>
        </TouchableOpacity>
      </View>
      <Modal transparent visible={pickerVisible}animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Choose page size</Text>
            {(['A4','LEGAL','LETTER'] as PageSizeName[]).map((size)=>(
              <TouchableOpacity key={size} style={styles.sizeOption} onPress={() => doExport(size)}>
                <Text style={styles.sizeOptionText}>{size}</Text>
              </TouchableOpacity>
            ))}
            <TouchableOpacity onPress={()=>setPickerVisible(false)}>
              <Text style={styles.cancelText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f2f2f2' },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 24 },
  emptyText: { textAlign: 'center', color: '#888', fontSize: 16 },
  bottomBar: { flexDirection: 'row', padding: 12, gap: 8, backgroundColor: '#fff' },
  scanButton: { flex: 1, backgroundColor: '#4CAF50', padding: 14, borderRadius: 8, alignItems: 'center' },
  exportButton: { flex: 1, backgroundColor: '#1976D2', padding: 14, borderRadius: 8, alignItems: 'center' },
  disabled: { opacity: 0.4 },
  buttonText: { color: '#fff', fontWeight: '600' },
  modalOverlay: { flex: 1, backgroundColor: '#00000066', justifyContent: 'flex-end' },
  modalCard: { backgroundColor: '#fff', padding: 20, borderTopLeftRadius: 16, borderTopRightRadius: 16 },
  modalTitle: { fontSize: 18, fontWeight: '600', marginBottom: 12 },
  sizeOption: { paddingVertical: 14, borderBottomWidth: 1, borderBottomColor: '#eee' },
  sizeOptionText: { fontSize: 16 },
  cancelText: { textAlign: 'center', color: '#D32F2F', marginTop: 12, fontSize: 16 },
});
