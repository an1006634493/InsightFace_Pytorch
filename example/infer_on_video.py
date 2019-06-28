import cv2
from PIL import Image
import argparse
from pathlib import Path
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank
import numpy as np

import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-f", "--file_name", help="video file name",default='video.mp4', type=str)
    parser.add_argument("-s", "--save_name", help="output file name",default='recording', type=str)
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)
    parser.add_argument("-d", "--duration", help="perform detection for how long(in seconds)", default=0, type=int)
    
    args = parser.parse_args()
    
    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')
    
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print(names)
        print('facebank updated')
        
    else:
        #pdb.set_trace() 
        targets, names = load_facebank(conf)
        print('facebank loaded')
        print(names)
        
    cap = cv2.VideoCapture(str(conf.facebank_path/args.file_name))
    
    cap.set(cv2.CAP_PROP_POS_MSEC, args.begin * 1000)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(str(conf.facebank_path/'{}.avi'.format(args.save_name)),
                                   cv2.VideoWriter_fourcc(*'XVID'), int(fps), (1280,720))
    
    if args.duration != 0:
        i = 0
    k = -1
    while cap.isOpened():
        k = k + 1
        
        isSuccess,frame = cap.read()
        
        #np.save("/hd1/anshengnan/InsightFace_Pytorch/data/frames/frame_ori_"+str(k)+".npy", frame)
        cv2.imwrite("/hd1/anshengnan/InsightFace_Pytorch/data/frame_imgs_ori/"+str(k).zfill(5)+".jpg", frame)
        if isSuccess:            
#             image = Image.fromarray(frame[...,::-1]) #bgr to rgb
            #pdb.set_trace() 
            image = Image.fromarray(frame)
            try:
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, 16)
                #pdb.set_trace() 
            except:
                bboxes = []
                faces = []
            if len(bboxes) == 0:
                print('no face')
                cv2.imwrite("/hd1/anshengnan/InsightFace_Pytorch/data/frame_imgs_det/"+str(k).zfill(5)+".jpg", frame)
                continue
            else:
                print('detect face')
                bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1] # personal choice   
                results, score = learner.infer(conf, faces, targets, True)
                
                for idx,bbox in enumerate(bboxes):
                    
                    if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                        #print(names[results[idx] + 1])
                        #print('_{:.2f}'.format(score[idx]))
                        #print(names[results[idx] + 1] + '_{:.2f}'.format(score[idx]))
                    else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
                        #print(names[results[idx] + 1])
            #np.save("/hd1/anshengnan/InsightFace_Pytorch/data/frames/frame_det_"+str(k)+".npy", frame)
            cv2.imwrite("/hd1/anshengnan/InsightFace_Pytorch/data/frame_imgs_det/"+str(k).zfill(5)+".jpg", frame)
            video_writer.write(frame)
        else:
            break
        if args.duration != 0:
            i += 1
            if i % 25 == 0:
                print('{} second'.format(i // 25))
            if i > 25 * args.duration:
                break        
    cap.release()
    video_writer.release()
    
