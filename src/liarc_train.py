from config import get_config
from liarc_Learner import face_learner
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=16, type=int)
    parser.add_argument("-net", "--net_mode", help="which network, [ir, ir_se, mobilefacenet,seesawFaceNet]",default='seesawFaceNet', type=str)
    parser.add_argument("-depth", "--net_depth", help="how many layers [50,100,152]", default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-1, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=196, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=8, type=int)
    parser.add_argument("-d", "--data_mode", help="use which database, [vgg, ms1m, emore, concat]",default='emore', type=str)
    args = parser.parse_args()

    conf = get_config()
    
    if args.net_mode == 'seesawFaceNet':
        conf.seesawFaceNet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth    
    
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    learner = face_learner(conf)

    learner.train(conf, args.epochs)
