"""Click treadmill BACK then FRONT, saving a 2D sign reference."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--time',type=float,default=0)
    args=parser.parse_args()
    cap=cv2.VideoCapture(str(args.input));cap.set(cv2.CAP_PROP_POS_MSEC,args.time*1000)
    ok,frame=cap.read();cap.release()
    if not ok:raise ValueError('Cannot read calibration frame')
    height,width=frame.shape[:2];factor=min(1,1100/width,800/height)
    preview=cv2.resize(frame,None,fx=factor,fy=factor);points=[]
    def click(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN and len(points)<2:points.append([x/factor,y/factor])
    cv2.namedWindow('Calibration');cv2.setMouseCallback('Calibration',click)
    try:
        while True:
            image=preview.copy()
            cv2.putText(image,'Click BACK then FRONT. Enter: save; R: reset; Esc: cancel',(12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            for p in points:cv2.circle(image,tuple((np.asarray(p)*factor).astype(int)),6,(0,255,255),-1)
            cv2.imshow('Calibration',image);key=cv2.waitKey(20)&255
            if key==27:return
            if key in [ord('r'),ord('R')]:points.clear()
            if key in [10,13] and len(points)==2:break
            if cv2.getWindowProperty('Calibration',cv2.WND_PROP_VISIBLE)<1:return
    finally:cv2.destroyAllWindows()
    direction=np.asarray(points[1])-points[0]
    if np.linalg.norm(direction)<5:raise ValueError('Calibration points are too close')
    direction/=np.linalg.norm(direction)
    config=dict(running_direction=direction.tolist(),back=points[0],front=points[1],
                image_size=[width,height],source_video=str(args.input.resolve()),
                role='2D direction sign reference; not a complete camera calibration')
    destination=args.output or args.input.parent/'camera_config.json'
    destination.write_text(json.dumps(config,indent=2),'utf-8');print(destination)


if __name__=='__main__':main()
