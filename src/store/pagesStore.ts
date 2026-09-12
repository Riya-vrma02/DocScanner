import { create } from 'zustand';
import { ScanPage } from '../types';

interface PagesState {
  pages: ScanPage[];
  addPage: (page: ScanPage) => void;
  updatePage: (id: string, patch: Partial<ScanPage>) => void;
  removePage: (id: string) => void;
  reorder: (newOrder: ScanPage[]) => void;
  clear: () => void;
}

/**
 * Holds the current scanning session's pages. Backing this with AsyncStorage/MMKV
 * (persist middleware) would let a document survive an app restart if needed.
 */
export const usePagesStore = create<PagesState>((set, get) => ({
  pages: [],
  addPage: (page) => set({ pages: [...get().pages, page] }),
  updatePage: (id, patch) =>
    set({
      pages: get().pages.map((p) => (p.id === id ? { ...p, ...patch } : p)),
    }),
  removePage: (id) => set({ pages: get().pages.filter((p) => p.id !== id) }),
  reorder: (newOrder) => set({ pages: newOrder }),
  clear: () => set({ pages: [] }),
}));
