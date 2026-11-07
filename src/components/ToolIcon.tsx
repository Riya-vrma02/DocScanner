import React from 'react';
import Svg, { Path, Rect, Circle, Polyline, Line } from 'react-native-svg';

export type ToolIconName =
  | 'retake' | 'crop' | 'rotate' | 'edit' | 'delete'
  | 'chevronLeft' | 'chevronRight' | 'addPage';

interface Props {
  name: ToolIconName;
  size?: number;
  color?: string;
}

/** Simple line icons drawn with react-native-svg (24x24 grid, stroke only). */
export default function ToolIcon({ name, size = 26, color = '#FFFFFF' }: Props) {
  const common = { stroke: color, strokeWidth: 1.8, fill: 'none', strokeLinecap: 'round' as const, strokeLinejoin: 'round' as const };
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      {name === 'retake' && (
        <>
          <Rect x="3" y="7" width="18" height="13" rx="2.5" {...common} />
          <Path d="M8.5 7 L10 4.5 H14 L15.5 7" {...common} />
          <Path d="M15.5 13.5 A3.5 3.5 0 1 1 13.8 10.5" {...common} />
          <Polyline points="14,8.8 14.3,10.9 12.2,11.2" {...common} />
        </>
      )}
      {name === 'crop' && (
        <>
          <Polyline points="6,2.5 6,18 21.5,18" {...common} />
          <Polyline points="2.5,6 18,6 18,21.5" {...common} />
        </>
      )}
      {name === 'rotate' && (
        <>
          <Path d="M19.5 12 A7.5 7.5 0 1 1 15.5 5.4" {...common} />
          <Polyline points="15.2,2.2 16,5.8 12.4,6.6" {...common} />
        </>
      )}
      {name === 'edit' && (
        <>
          <Line x1="4" y1="7" x2="20" y2="7" {...common} />
          <Line x1="4" y1="12" x2="20" y2="12" {...common} />
          <Line x1="4" y1="17" x2="20" y2="17" {...common} />
          <Circle cx="9" cy="7" r="2" {...common} fill="#1F2125" />
          <Circle cx="15" cy="12" r="2" {...common} fill="#1F2125" />
          <Circle cx="7" cy="17" r="2" {...common} fill="#1F2125" />
        </>
      )}
      {name === 'delete' && (
        <>
          <Line x1="4" y1="6.5" x2="20" y2="6.5" {...common} />
          <Path d="M9 6.5 V4.5 H15 V6.5" {...common} />
          <Path d="M6.5 6.5 L7.5 20 H16.5 L17.5 6.5" {...common} />
          <Line x1="10.5" y1="10.5" x2="10.5" y2="16.5" {...common} />
          <Line x1="13.5" y1="10.5" x2="13.5" y2="16.5" {...common} />
        </>
      )}
      {name === 'chevronLeft' && <Polyline points="15,5 8,12 15,19" {...common} strokeWidth={2.4} />}
      {name === 'chevronRight' && <Polyline points="9,5 16,12 9,19" {...common} strokeWidth={2.4} />}
      {name === 'addPage' && (
        <>
          <Rect x="4" y="3" width="12" height="16" rx="1.5" {...common} />
          <Line x1="17" y1="14" x2="17" y2="22" {...common} />
          <Line x1="13" y1="18" x2="21" y2="18" {...common} />
        </>
      )}
    </Svg>
  );
}