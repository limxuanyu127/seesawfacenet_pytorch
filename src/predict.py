import argparse
from PIL import Image
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
import cv2

from utils import load_facebank, draw_box_name, prepare_facebank

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save",action="store_true")
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true")

    parser.add_argument("-p", "--path", help="path to read image for prediction",action="store_true", required=True)
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')
    
    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        #TODO change the file path
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    #Load face bank
    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta = args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    #Read Image
    img_path = args.path
    img = cv2.imread(img_path)

    #predict
    try:
        #Convert to Pillow image
        image = Image.fromarray(img)
        bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
        bboxes = bboxes[:,:-1] #shape:[10,4],only keep 10 highest possibiity faces
        bboxes = bboxes.astype(int)
        bboxes = bboxes + [-1,-1,1,1] # personal choice    
        results, score = learner.infer(conf, faces, targets, args.tta)

        # for idx,bbox in enumerate(bboxes):
        #     if args.score:
        #         frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
        #     else:
        #         frame = draw_box_name(bbox, names[results[idx] + 1], frame)
    except:
        print('detect error')

    
    #print the frame
    print('results',results)
    print('score', score)