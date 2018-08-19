# coding=utf-8
import tensorflow as tf
import os
import pickle
import argparse
import cv2
import numpy as np
import json
import scipy.misc
import util
from scipy.misc import imread
from tqdm import tqdm  # tqdm是一个终端进度条工具
from util import crop, get_transform, flipRef, group, kpt_affine, adjustKeypoint, draw_limbs
from model import inference
from tensorflow.contrib.framework import assign_from_checkpoint_fn

def loadNetwork(path, sess, model_name):
    img = tf.placeholder(dtype = tf.float32, shape = (None, None, None, 3))
    with tf.variable_scope(model_name):
        pred = inference(img, 68 if model_name=='my_model' else 17)  # 为什么是68，github上作者解释过了，模型是68，但是实际使用了前34（论文中提到是34）

    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # 设置Tensorboard
    writer=tf.summary.FileWriter("TensorBoard",sess.graph)

    variables_to_restore = tf.global_variables()
    dic = {}
    for i in variables_to_restore:
        if 'global_step' not in i.name and 'Adam' not in i.name:
            dic[str(i.op.name).replace(model_name+'/', 'my_model/')] = i
    init_fn = assign_from_checkpoint_fn(os.path.join(path, 'snapshot'), dic, ignore_missing_vars = True)
    init_fn(sess)

    def func(imgs):
        output = sess.run(pred, feed_dict={img: imgs})
        print('-------output.shape-------')
        print(output.shape) # (2, 200, 200, 68)
        return {
            'det': output[:,:,:,:17], # 前17个 #detection scores
            'tag': output[:,:,:,-17:] # 后17个 #identity tags
        }
    return func

def parse_args():
    parser = argparse.ArgumentParser()
    #which_args = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, default='multiperson', help='multiperson model path')
    parser.add_argument('-r', '--refine_model_path', type=str, default=None, help='refinement model path', choices=[None, 'refinement'])
    parser.add_argument('--scales', type=str, default='single', help='switch for multi-scale evaluation', choices=['multi', 'single'])

    parser.add_argument('-i', '--input_image_path', type=str,default='visionteam1.jpg', help='input image name')
    parser.add_argument('-o', '--output_image_path', type=str, default='output.jpg', help='output image name')

    parser.add_argument('-l', '--imglist', type=str, default=None, help='image path list')
    parser.add_argument('-f', '--output_file', type=str, default='output.json', help='prediction saving files')
    args = parser.parse_args()
    return args

def resize(im, res):
    return np.array([cv2.resize(im[:,:,i],res) for i in range(im.shape[2])]).transpose(1, 2, 0)

def multiperson(img, func, mode):
    if mode == 'multi':
        scales = [2., 1.8, 1.5, 1.3, 1.2, 1., 0.7, 0.5, 0.25]
    else:
        scales = [1]

    height, width = img.shape[0:2]  # 582,800； 这里假设height,width中最大的为800，以下的shape都是依据800计算的
    center = (width/2, height/2)
    dets, tags = None, []
    for idx, i in enumerate(scales):
        scale = max(height, width)/200
        inp_res = int((i * max(512, max(height, width)) + 3)//4 * 4)
        res = (inp_res, inp_res)  # 800,800
        inp = crop(img, center, scale, res)

        tmp = func([inp, inp[:,::-1]])
        det = tmp['det'][0,:,:] + tmp['det'][1,:,::-1][:,:,flipRef] #det shape [200,200,17]
        if idx == 0:
            dets = det
            mat = get_transform(center, scale, res)[:2]  # shape:[2,3] ;mat=[1 0 0;0 1 109],表示没有缩放，只有y轴方向偏移109=(800-582)/2
            mat = np.linalg.pinv(np.array(mat).tolist() + [[0,0,1]])[:2] # 计算仿射变换矩阵mat的伪逆
        else:
            dets = dets + resize(det, dets.shape[0:2]) 

        if abs(i-1)<0.5:
            res = dets.shape[0:2]  # res shape:[200,200]
            tags += [resize(tmp['tag'][0,:,:,:], res), resize(tmp['tag'][1,:,::-1][:,:,flipRef], res)]

    tags = np.concatenate([i[:,:,:,None] for i in tags], axis=3)  #tags shape:[200,200,17,2]
    dets = dets/len(scales)/2
    import cv2
    cv2.imwrite('det.jpg', (tags.mean(axis=3).mean(axis=2) *255).astype(np.uint8))
    grouped = group(dets, tags, 30)  
    # grouped shape:[30,17,5] 表示检测到30个人，每个人17个关键点，每个关键点前三个数表示x，y，prediction
    grouped[:,:,:2] = kpt_affine(grouped[:,:,:2], mat) 

    # 筛选并整合人体关键点信息
    persons = []
    for val in grouped: # val为某一个人的关键点信息
        if val[:, 2].max()>0: # 某个人的17个关键点中最大的prediction必须大于0
            tmp = {"keypoints": [], "score":float(val[:, 2].mean())}  # 将17个关键点的平均值作为score分数值
            for j in val:  # j表示17个关键点中的某一个
                if j[2]>0.: # 关键点的prediction必须大于0，否则认为检测错误，记为[0,0,0]
                    tmp["keypoints"]+=[float(j[0]), float(j[1]), float(j[2])]
                else:
                    tmp["keypoints"]+=[0, 0, 0]
            persons.append(tmp)
    return persons # 返回满足要求的所有人

def refinement(img, pred_dict, func):
    for idx, person in enumerate(pred_dict):
        pred = np.array( person['keypoints'] ).reshape(-1, 3)

        pts = np.array(pred).reshape(-1, 3)
        bbox = pts[pts[:, 2]>0]
        x1, x2, y1, y2 = bbox[:, 0].min(), bbox[:, 0].max(), bbox[:, 1].min(), bbox[:, 1].max()
        scale = max(max(x2 -x1, y2 -y1), 60) * 1.3/200
        center = (x2 + x1)/2, (y1 + y2)/2.

        res = (512, 512)
        det = []
        for res_idx, res in enumerate([(512, 512), (400, 400), (300, 300)]):
            inp = crop(img, center, scale, res)
            if res_idx == 0:
                mat = np.linalg.inv(get_transform(center, scale, res))[:2]

            tmp = func([inp, inp[:,::-1]])
            det.append( (tmp['det'][0,:,:] + tmp['det'][1,:,::-1][:,:,flipRef])/2 )
        det = np.mean([resize(i, det[0].shape[:2]) for i in det], axis = 0)

        old = pred.copy()
        match_keypoint = []
        for i in range(17):
            tmp = det[:,:,i]
            x, y = np.unravel_index( np.argmax(tmp), tmp.shape )
            val = tmp.max()
            x, y = kpt_affine( adjustKeypoint(tmp, ([x],[y]))[:,::-1] * 4, mat )[0]
            if pred[i, 2] > 0:
                match_keypoint.append(np.sum((pred[i, :2] - np.array((x, y)))**2) < 100)
            if pred[i, 2] ==0 or val >0.15:
                pred[i, :3] = (x, y, val)
        if np.mean(match_keypoint) < 0.2:
            pred = old
        pred_dict[idx]['keypoints'] = pred.reshape(-1).tolist()
    return pred_dict


def main():
    opt = parse_args()
    # tf.ConfigProto对Session进行参数设置
    # 如果你指定的设备不存在，允许TF自动分配设备
    config = tf.ConfigProto(allow_soft_placement=True)#, log_device_placement= True)
    # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
    # 内存，所以会导致碎片
    config.gpu_options.allow_growth=True
    sess = tf.Session(config = config)
    util.sess = sess

    func = loadNetwork(opt.model_path, sess, 'my_model')
    if opt.refine_model_path != None:
        sess2 = tf.Session(config = config)
        refine_func = loadNetwork(opt.refine_model_path, sess2, 'refine')

    # 如果读取的图片列表不为空，则生成图片list
    if opt.imglist is not None: 
        with open(opt.imglist, 'r') as f:
            imgfiles = f.readlines()
    else:
        imgfiles = [opt.input_image_path]

    preds = []  # 检测得到的关键点(每个人17个关键点)，关键点坐标x,y,prediction
    for img_path in tqdm(imgfiles, total = len(imgfiles)):
        img = imread(img_path.strip(), mode='RGB')
        people = multiperson(img, func, opt.scales)
        if opt.refine_model_path != None:
            people = refinement(img, people, refine_func)
        for i in people:
            i['image_path'] = img_path.strip()
        preds.append(people)
    print("------Detection Keypoints--------")
    print(preds[0])

    if opt.output_file is not None:
        if opt.output_file[-4:] == 'json':
            with open(opt.output_file, 'w') as f:
                json.dump(preds, f)
        else:
            with open(opt.output_file, 'wb') as f:
                pickle.dump(preds, f)

    if opt.output_image_path is not None:
        img = imread(imgfiles[0].strip(), mode='RGB')
        for i in preds[0]:
            draw_limbs(img, i['keypoints']) # 绘制人体关键点连线，只有当关键点的可能性高于0.07时，才连接关键点
        cv2.imwrite(opt.output_image_path, img[:,:,::-1]) #opencv中RGB保存为BGR顺序，即倒序排列

if __name__ == '__main__':
    main()