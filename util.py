# coding=utf-8
import tensorflow as tf
from model import pool
from munkres import Munkres  # 二分图最佳匹配算法 kuhn munkras算法 匈牙利算法
import cv2
import scipy.misc
import numpy as np

flipRef = [i-1 for i in [1,3,2,5,4,7,6,9,8,11,10,13,12,15,14,17,16] ] # 翻转情况下左右肢体名称序号交换

# 一共17个关键点
part_labels = ['nose','eye_l','eye_r','ear_l','ear_r',
               'sho_l','sho_r','elb_l','elb_r','wri_l','wri_r',
               'hip_l','hip_r','kne_l','kne_r','ank_l','ank_r']
part_idx = {b:a for a, b in enumerate(part_labels)}  # 生成字典'nose':0等

def draw_limbs(inp, pred):
    """
    inp:input image
    pred:检测到的人体关键点，坐标x1,y1,prediction1,x2,y2,prediction2...x17,y17,prediction17
    """
    def link(a, b, color):
        """设置关键点连线的属性"""
        if part_idx[a] < pred.shape[0] and part_idx[b] < pred.shape[0]:
            a = pred[part_idx[a]]
            b = pred[part_idx[b]]
            # a,b 为某个关键点的信息，包括x,y,prediction
            if a[2]>0.07 and b[2]>0.07: # 只有当关键点的可能性高于0.07时，才连接关键点
                cv2.line(inp, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), color, 6) # 线条粗细为6

    pred = np.array(pred).reshape(-1, 3)
    bbox = pred[pred[:,2]>0]
    a, b, c, d = bbox[:,0].min(), bbox[:,1].min(), bbox[:,0].max(), bbox[:,1].max()

    # 绘制一个包括17个人体关键点的最小边框，表示检测到的某个人
    cv2.rectangle(inp, (int(a), int(b)), (int(c), int(d)), (255, 255, 255), 2) # 白色边框

    # 这里定义了，17个关键点之间如何连接，以及线条颜色
    link('nose', 'eye_l', (255, 0, 0))
    link('eye_l', 'eye_r', (255, 0, 0))
    link('eye_r', 'nose', (255, 0, 0))

    link('eye_l', 'ear_l', (255, 0, 0))
    link('eye_r', 'ear_r', (255, 0, 0))

    link('ear_l', 'sho_l', (255, 0, 0))
    link('ear_r', 'sho_r', (255, 0, 0))
    link('sho_l', 'sho_r', (255, 0, 0))
    link('sho_l', 'hip_l', (0, 255, 0))
    link('sho_r', 'hip_r',(0, 255, 0))
    link('hip_l', 'hip_r', (0, 255, 0))

    link('sho_l', 'elb_l', (0, 0, 255))
    link('elb_l', 'wri_l', (0, 0, 255))

    link('sho_r', 'elb_r', (0, 0, 255))
    link('elb_r', 'wri_r', (0, 0, 255))

    link('hip_l', 'kne_l', (255, 255, 0))
    link('kne_l', 'ank_l', (255, 255, 0))

    link('hip_r', 'kne_r', (255, 255, 0))
    link('kne_r', 'ank_r', (255, 255, 0))

def py_max_match(scores):
    m = Munkres()
    tmp = m.compute(-scores)
    tmp = np.array(tmp).astype(np.int32)
    return tmp

def match_by_tag(inp, pad=True):
    '''匹配关键点，暂时没看懂'''
    tag_k, loc_k, val_k = inp
    default_ = np.zeros((17, 3 + tag_k.shape[2])) # 每个关键点的x,y之后增加三个数； [17,5]

    dic = {}
    dic2 = {}
    for i in [1,2,3,4,5,6,7,12,13,8,9,10,11,14,15,16,17]:
        ptIdx = i-1
        tags = tag_k[ptIdx]
        joints = np.concatenate((loc_k[ptIdx], val_k[ptIdx, :, None], tags), 1)  # x,y,prediction,tags-1,tags-2;每个关键点五个值
        mask = joints[:, 2] > 0.03  # 筛选作用，选择prediction>0.03的关键点
        tags = tags[mask]
        joints = joints[mask] #实际上，到这里已经把所有人体检测到了，一下主要是人体的关键点匹配
        if i == 0 or len(dic) == 0: # 当i等于0时，检测到了所有人体的鼻子，设置为字典开头
            for tag, joint in zip(tags, joints):
                dic.setdefault(tag[0], np.copy(default_))[ptIdx] = joint # 将dic设置为tag[0]:17x5的数组
                dic2[tag[0]] = [tag]  # dic2设置为tag[0]:tag
        else:
            actualTags = list(dic.keys())[:30]
            actualTags_key = actualTags
            actualTags = [np.mean(dic2[i], axis = 0) for i in actualTags]

            diff = ((joints[:, None, 3:] - np.array(actualTags)[None, :, :])**2).mean(axis = 2) ** 0.5
            if diff.shape[0]==0:
                continue

            diff2 = np.copy(diff)
            diff = np.round(diff) * 100 - joints[:, 2:3]
            if diff.shape[0]>diff.shape[1]:
                diff = np.concatenate((diff, np.zeros((diff.shape[0], diff.shape[0] - diff.shape[1])) + 1e10), axis = 1)

            pairs = py_max_match(-diff)
            for row, col in pairs:
                if row<diff2.shape[0] and col < diff2.shape[1] and diff2[row][col] < 1:
                    dic[actualTags_key[col]][ptIdx] = joints[row]
                    dic2[actualTags_key[col]].append(tags[row])
                else: 
                    key = tags[row][0]
                    dic.setdefault(key, np.copy(default_))[ptIdx] = joints[row] # dic添加新的人体模型
                    dic2[key] = [tags[row]]

    ans = np.array([dic[i] for i in dic])
    if pad:  # 填充为30个人体模型（因为输入参数K=30,所以需要输出30个）
        num = len(ans)
        if num < 30:
            padding = np.zeros((30-num, 17, default_.shape[1]))
            if num>0: ans = np.concatenate((ans, padding), axis = 0)
            else: ans = padding
        return np.array(ans[:30]).astype(np.float32)
    else:
        return np.array(ans).astype(np.float32)

def nms_top_k(inps, K=30):
    inp, maxes = inps

    eq = tf.equal(inp, maxes)
    mt = tf.to_float(eq) * inp
    locs = tf.where(tf.greater(mt, 0))
    vals = tf.gather_nd(mt, locs)
    val_k, loc_k = tf.nn.top_k(vals, k = tf.minimum( tf.shape(locs)[0], K ), sorted = True)
    loc_k = tf.gather(locs, loc_k)
    length = tf.size(val_k) 
    tmp = lambda: tf.pad(loc_k, [[0, K - length], [0,0]])
    loc_k = tf.cond( length<K, tmp, lambda: loc_k)[:K]
    return tf.to_int64(tf.reshape(loc_k, tf.constant((K, 2))))

def make_top_k_func():
    K=30
    input = tf.placeholder(dtype = np.float32, shape = (None, None, None, None))
    #input = tf.placeholder(dtype = np.float32, shape = (1, 128, 128, 17))
    nms = pool(input, 'nms_max_pool', [3, 3], [1, 1])

    nms = tf.transpose(nms, (0, 3, 1, 2))
    hm = tf.transpose(input, (0, 3, 1, 2))
    shape =  tf.shape(hm)
    hm = tf.reshape(hm, (shape[0]*shape[1], shape[2], shape[3]))
    nms = tf.reshape(nms, tf.shape(hm))

    results = tf.map_fn(lambda x:nms_top_k(x, K), (hm, nms), tf.int64)
    results = tf.reshape(results, (shape[0], shape[1], K, 2))

    def func(inp):
        return sess.run(results, feed_dict={input:inp[None, :,:,:]})[0]
    return func

top_k_func = make_top_k_func()

def adjustKeypoint(tmp, loc):
    ans = []
    for x, y in zip(*loc):
        xx, yy = x, y
        if tmp[xx, min(yy+1, tmp.shape[1]-1)]>tmp[xx, max(yy-1, 0)]:
            y+=0.25
        else:
            y-=0.25

        if tmp[min(xx+1, tmp.shape[0]-1), yy]>tmp[max(0, xx-1), yy]:
            x+=0.25
        else:
            x-=0.25
        ans.append((x + 0.5, y + 0.5))
    return np.array(ans)

def group(det, tag, K=30):
    '''
    det.shape=[200,200,17]
    tag.shape=[200,200,17,2]
    '''
    loc_k, tag_k, val_k = [], [], []
    top_k = top_k_func(det)  # 返回17类关键点的prediction前30个坐标位置；top_k.shape=[17,30,2]
    for i in range(17):
        #tmp_loc = np.unravel_index( np.argsort(-(tmp*(NMS(tmp)==tmp)).reshape(-1))[:K], tmp.shape )
        tmp = det[:,:,i]
        tmp_loc = top_k[i].transpose(1, 0).tolist()
        tag_k.append( tag[:,:,i][tmp_loc] )
        val_k.append(det[:,:,i][tmp_loc])
        loc_k.append(adjustKeypoint(tmp, tmp_loc)[:,::-1] * 4)
    return match_by_tag((np.array(tag_k), np.array(loc_k), np.array(val_k)))

def resize(*args):
    im = args[0]
    if im.ndim == 3 and im.shape[2] > 3:
        res = args[1]
        new_im = np.zeros((res[0], res[1], im.shape[2]), np.float32)
        for i in range(im.shape[2]):
            if im[:,:,i].max() > 0:
                new_im[:,:,i] = resize(im[:,:,i], res, *args[2:])
        return new_im
    else:
        return scipy.misc.imresize(*args).astype(np.float32) / 255

def get_transform(center, scale, res, rot=0):
    # Generate transformation matrix
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + .5)
    t[1, 2] = res[0] * (-float(center[1]) / h + .5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot # To match direction of rotation from cropping
        rot_mat = np.zeros((3,3))
        rot_rad = rot * np.pi / 180
        sn,cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0,:2] = [cs, -sn]
        rot_mat[1,:2] = [sn, cs]
        rot_mat[2,2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0,2] = -res[1]/2
        t_mat[1,2] = -res[0]/2
        t_inv = t_mat.copy()
        t_inv[:2,2] *= -1
        t = np.dot(t_inv,np.dot(rot_mat,np.dot(t_mat,t)))
    return t

def transform(pt, center, scale, res, invert=0, rot=0):
    # Transform pixel location to different reference
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2].astype(int)

def crop(img, center, scale, res, rot=0):
    # Upper left point
    ul = np.array(transform([0, 0], center, scale, res, invert=1))
    # Bottom right point
    br = np.array(transform(res, center, scale, res, invert=1))

    # Padding so that when rotated proper amount of context is included
    pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2)
    if not rot == 0:
        ul -= pad
        br += pad

    new_shape = [br[1] - ul[1], br[0] - ul[0]]
    if len(img.shape) > 2:
        new_shape += [img.shape[2]]

    new_img = np.zeros(new_shape)

    # Range to fill new array
    new_x = max(0, -ul[0]), min(br[0], len(img[0])) - ul[0]
    new_y = max(0, -ul[1]), min(br[1], len(img)) - ul[1]
    # Range to sample from original image
    old_x = max(0, ul[0]), min(len(img[0]), br[0])
    old_y = max(0, ul[1]), min(len(img), br[1])
    tmp = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
    #print(old_y[0],old_y[1], old_x[0],old_x[1])
    #print(new_y[0], new_y[1], new_x[0], new_x[1], tmp.max())
    if old_x[0]<old_x[1] and old_y[0] < old_y[1]:
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = tmp

    if not rot == 0:
        # Remove padding
        # something is very stupid that it would convert 1. to 255 here
        new_img = scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    #print(img.max(), tmp.max(), new_img.max(), new_img.astype(np.uint8).max())
    return resize(new_img.astype(np.uint8), res)

def kpt_affine(kpt, mat):
    '''关键点仿射变换，变换矩阵为mat'''
    kpt = np.array(kpt)
    shape = kpt.shape # 缓存kpt原来的shape，最后一步恢复为原来的shape
    kpt = kpt.reshape(-1, 2) # kpt shape=[?,2]；每行为x,y
    # 转换为齐次坐标形式,kpt每行变为x,y,1；mat.T的shape=[3,2]
    return np.dot( np.concatenate((kpt, kpt[:, 0:1]*0+1), axis = 1), mat.T ).reshape(shape)
