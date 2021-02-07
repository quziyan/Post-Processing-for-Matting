import cv2,os
import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
import scipy as sp
import scipy.optimize as opt
import time

def cv2_imread(file_path, toRGB = False, max_border_len=None, shape=None):
    
    #cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    cv_img = cv2.imread(file_path)
    if toRGB:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    assert not (max_border_len is not None and shape is not None)
    if max_border_len is not None:
        h, w = cv_img.shape[0:2]
        ratio =  max_border_len / max(h, w)
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        cv_img = cv2.resize(cv_img, (new_w, new_h))
    if shape is not None:
        h, w = shape
        cv_img = cv2.resize(cv_img, (w, h))

    return cv_img
def cv2_imwrite( path, img, toBGR=False):
    suffix = os.path.splitext(path)[-1]
    if toBGR:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imencode(suffix, img)[1].tofile(path)
def dilation(mask, ksize=20):
    kernel = np.ones((ksize, ksize),np.float)
    mask_di = cv2.dilate(mask, kernel, )
    mask_di = mask_di[:,:,np.newaxis]
    return mask_di

def get_similarity(vec1, vec2, similarity_type, space ):
        #similarity_type l2/cosine
        #space rgb/hsv/
        if space == 'hsv':
            vec1 = cv2.cvtColor(vec1, cv2.COLOR_RGB2HSV)[:,:, 2]
            vec2 = cv2.cvtColor(vec2, cv2.COLOR_RGB2HSV)[:,:, 2]
        vec1 = vec1 / 255.0
        vec2 = vec2 / 255.0
        
        similarity = 0
        if similarity_type == 'cosine':
            vec1 = np.linalg.norm(vec1)
            vec2 = np.linalg.norm(vec2)
            similarity = np.dot(vec1, vec2.T)/(vec1 * vec2)
        elif similarity_type == 'l2':
            dist = ((vec1 - vec2)**2).mean() ** 0.5
            similarity = 1 - dist
        elif similarity_type == 'l1':
            dist = (np.abs(vec1 - vec2)).mean()
            similarity = 1 - dist
        return similarity

def get_similarity_vec(vec1, vec2, similarity_type, space ):
        #similarity_type l2/cosine
        #space rgb/hsv/
        if space == 'hsv':
            vec1 = cv2.cvtColor(vec1, cv2.COLOR_RGB2HSV)[:,:, 2:]
            vec2 = cv2.cvtColor(vec2, cv2.COLOR_RGB2HSV)[:,:, 2:]
        vec1 = vec1 / 255.0
        vec2 = vec2 / 255.0
        
        similarity = 0
        eps = 1e-6
        if similarity_type == 'cosine':
            vec1_norm = np.linalg.norm(vec1, axis=2, keepdims=False)
            vec2_norm = np.linalg.norm(vec2, axis=2, keepdims=False)
            similarity = (vec1 * vec2).sum(axis=2, keepdims=False) / (vec1_norm * vec2_norm + eps)
        elif similarity_type == 'l2':
            dist = ((vec1 - vec2)**2).mean(axis=2, keepdims=False) ** 0.5
            similarity = 1 - dist
        elif similarity_type == 'l1':
            dist = (np.abs(vec1 - vec2)).mean(axis=2, keepdims=False)
            similarity = 1 - dist
        return similarity    

def get_win_centerdist_similarity(mask_win, center_h, center_w):
    #计算window与中心点的位置归一化相似度（越近，相似度越高)
    ksize_h, ksize_w= mask_win.shape
    #r = max(ksize_h - center_i, ksize_w - ) // 2
    r_h = max(center_h, ksize_h - 1 - center_h)
    r_w = max(center_w, ksize_w - 1 - center_w)
    r= max(r_h, r_w)
    maxdist = (r ** 2 * 2) ** 0.5
    similarity = np.zeros_like(mask_win)
    for i in range(ksize_h):
        for j in range(ksize_w):
            ijdist = ((i - center_h) ** 2 + (j - center_w) ** 2) ** 0.5
            similarity[i , j] = (1-ijdist) / maxdist
    return similarity


def get_max(image_ori, image, mask, ksize, similarity_type, space, evaluate_eps = 0.00):
    #image_ori 原始图
    #image 增强后的图
    image_re = np.copy(image)
    def get_mask_max(mask, i, j, ksize):
        h, w = mask.shape
        r = ksize // 2
        h_start = max(i - r, 0)
        h_end = min(h-1, i+r)
        w_start = max(j - r, 0)
        w_end = min(w-1, j+r)
        max_val = -1
        h_max = -1
        w_max = -1

        image_win = image[h_start:h_end, w_start:w_end]
        mask_win = mask[h_start:h_end, w_start:w_end]
        similarity = mask_win + \
            0.01*get_similarity_vec(image_win, image_ori[i:i+1,j:j+1], similarity_type = similarity_type, space=space) + \
                0 #0.01 * get_win_centerdist_similarity(mask_win=mask_win, center_h=i-h_start, center_w=j-w_start)
        pos = np.unravel_index(np.argmax(similarity),similarity.shape)

        h_max = h_start + pos[0]
        w_max = w_start + pos[1]
        return h_max, w_max
        '''
        for h_ in range(h_start, h_end):
            for w_ in range(w_start, w_end):
                similarity = mask[h_, w_] + \
                                    get_similarity(image[h_:h_+1, w_:w_+1], 
                                        image[i:i+1,j:j+1], 
                                        similarity_type = similarity_type,
                                        space=space)
                if similarity > max_val:
                    max_val = similarity
                    h_max = h_
                    w_max = w_
        return h_max, w_max
        '''

    h, w = mask.shape
    for i in range(h):
        for j in range(w):
            alpha_ij = mask[i, j]
            if alpha_ij <= evaluate_eps and alpha_ij > 0.0: # alpha_ij > 0 + evaluate_eps and alpha_ij < 1 - evaluate_eps:
                i_max, j_max = get_mask_max(mask, i, j, ksize) 
                image_re[i, j] = image[i_max, j_max]
            #else:
    return image_re




def get_image_bg(image_ori, mask, ksize, evaluate_eps=0.00, calculate_method='lsq'):
    #估计alpha在(0,1)范围内的背景
    def linear_strech(array, start, end):
        #这个函数不好用
        to_min_val = max(min(array), start)
        to_max_val = min(end, max(array))
        result = (array - min(array)) / (max(array) - min(array)) * (to_max_val - to_min_val) + to_min_val
        return result


    image_bg = np.copy(image_ori)
    image_fg = np.copy(image_ori)
    h, w = mask.shape
    r = ksize // 2
    #dist_weight = get_dist_weight(ksize).reshape(-1, 1)
    for i in range(h):
        #print(i)
        for j in range(w):
            if mask[i,j] > 0 + evaluate_eps and mask[i,j] < 1 - evaluate_eps:
                h_start = max(i - r, 0)
                h_end = min(h-1, i+r)
                w_start = max(j - r, 0)
                w_end = min(w-1, j+r)
                mask_win = mask[h_start:h_end, w_start: w_end]
                image_ori_win = image_ori[h_start:h_end, w_start: w_end]
                ''' #Navie实现背景估计
                bg_win = image_ori_win[mask_win ==0]
                if bg_win.size == 0:
                    pos = np.unravel_index(np.argmin(mask_win), mask_win.shape)
                    image_bg[i,j] = image_ori[pos[0], pos[1]]
                    #image_bg[i,j] = image_ori[i,j]
                else:
                    image_bg[i,j] =  bg_win.mean(axis=0, keepdims=True)
                '''
                dist_weight = get_win_centerdist_similarity(mask_win, center_h=i-h_start, center_w=j-w_start).reshape(-1, 1)
                #最小二乘实现背景估计
                mask_win_vec_raw = mask_win.reshape(-1, 1)
                mask_win_vec = np.concatenate((mask_win_vec_raw, 1-mask_win_vec_raw), axis=1) #alpha, 1-alpha N*2
                image_ori_win_vec = image_ori_win.reshape(-1, 3) # N*3
                if calculate_method == 'lsq':
                    #这部分假设过强，假设前景一致，背景一致
                    #print(mask_win_vec.shape, dist_weight.shape)
                    mask_win_vec = mask_win_vec * dist_weight #mask_win_vec_raw * dist_weight
                    image_ori_win_vec = image_ori_win_vec * dist_weight #image_ori_win_vec * dist_weight
                    #print(mask_win_vec.shape, image_ori_win_vec.shape)
                    #fg_bg 2*3
                    fg_bg, _res, rank, singular = np.linalg.lstsq(mask_win_vec, image_ori_win_vec)
                    #print(fg_bg)
                    #print(fg_bg)
                    image_fg[i,j] = fg_bg[0].clip(0,255) #linear_strech(fg_bg[0], 0, 255) #fg_bg[0].clip(0,255)
                    image_bg[i,j] = fg_bg[1].clip(0,255) #linear_strech(fg_bg[1], 0, 255) #fg_bg[1].clip(0,255)
                elif calculate_method == 'naive':
                    bg_win = image_ori_win[mask_win ==0]
                    if bg_win.size == 0:
                        pos = np.unravel_index(np.argmin(mask_win), mask_win.shape)
                        image_bg[i,j] = image_ori[pos[0], pos[1]]
                        #image_bg[i,j] = image_ori[i,j]
                    else:
                        image_bg[i,j] =  bg_win.mean(axis=0, keepdims=True)
                    mask_ij = mask[i, j] if mask[i,j] >= 0.01 else 1.0
                    image_fg[i, j] = ( image_ori[i, j] - ( 1 - mask_ij ) ) / mask_ij
                else:
                    raise


            else:
                image_fg[i,j] = image_ori[i,j]
                image_bg[i,j] = image_ori[i,j]
    return image_bg, image_fg


if __name__ == '__main__':
    DEBUG=True
    def get_blue_bg(refimg):
        bg = np.zeros_like(refimg)
        bg[:,:, 2] = 255
        return bg
    def text_on_img_bottom(img, text):
        h, w, c  = img.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        re = cv2.putText(img, text, (50,50), font, 0.8, (255, 0, 0), 2)
        return re

    def process(img_ori, mask, blue_bg, img_pixel, imgoutput_path):
        #subplot = lambda i, j: plt.subplot(rows, cols, (i)* cols + j + 1)
        img_ori_bg, img_ori_fg = get_image_bg(img_ori, mask, ksize=10, evaluate_eps=0.05, calculate_method='lsq')
        matting = lambda foreground: (foreground * mask[:,:, np.newaxis] + blue_bg * (1-mask[:,:, np.newaxis])).astype(np.uint8)
        image_ori_fgenhanced = img_ori_fg # get_enhanced_image(img_ori, img_ori_bg, mask, mask_gamma_coeff=1.0)

        image_ori_fgenhanced_final = \
            get_max(img_ori, image_ori_fgenhanced, mask, ksize=35, similarity_type='l1', space='hsv', evaluate_eps=0.05)

        matting_img_ori = matting(img_ori)
        matting_img_pixel = matting(img_pixel)
        matting_img_fgenhanced = matting(image_ori_fgenhanced_final)

        if DEBUG:
            output = np.concatenate((
                np.repeat((mask[:,:,np.newaxis] * 255).astype(np.uint8), 3, axis=2), 
                text_on_img_bottom(img_ori, 'ORI_IMG'), 
                text_on_img_bottom(img_ori_bg, 'MID_BG'),
                text_on_img_bottom(image_ori_fgenhanced, 'MID_FG'),
                text_on_img_bottom(img_pixel, 'ORI_Pixel'),
                text_on_img_bottom(image_ori_fgenhanced_final, 'MID_FG_ENC'), 
                text_on_img_bottom(matting_img_ori, 'MATTING_ORI_IMAGE'),
                text_on_img_bottom(matting_img_pixel, 'MATTING_ORI_PIXEL'),
                text_on_img_bottom(matting_img_fgenhanced, 'MATTING_FG_ENC'), 
                ),axis=1)
            cv2_imwrite(imgoutput_path, output, toBGR=True)
        else:
            output = np.concatenate((
                np.repeat((mask[:,:,np.newaxis] * 255).astype(np.uint8), 3, axis=2), 
                text_on_img_bottom(img_ori, 'ORI_IMG'), 
                text_on_img_bottom(img_pixel, 'ORI_Pixel'),
                text_on_img_bottom(image_ori_fgenhanced_final, 'MID_PIXEL'), 
                text_on_img_bottom(matting_img_pixel, 'MATTING_ORI_IMAGE'),
                text_on_img_bottom(matting_img_pixel, 'MATTING_ORI_PIXEL'),
                text_on_img_bottom(matting_img_fgenhanced, 'MATTING_MID_PIXEL'), 
                ),axis=1)
            cv2_imwrite(imgoutput_path, output, toBGR=True)

    def get_path(pathbase, pathtype):
        if pathtype == 'mask':
            re0 =pathbase.replace('-a', '-b')
        if pathtype == 'pixel':
            re0 =pathbase.replace('-a', '-c')
        if pathtype == 'output':
            re0 = pathbase.replace('-a', '-d')
        rawpath, ext = os.path.splitext(re0)
        re1 = rawpath + '.png'
        return re1
        raise
            

    #image_ORI_paths = glob.glob(r'D:\Data\Download\倍赛提供的数据\SWB00320-a.*')
    #image_ORI_paths = glob.glob(r'D:\Data\Download\倍赛提供的数据\*-a.*')
    #image_ORI_paths = glob.glob(r'D:\Data\Download\to曲直\新整理数据\证件照测试集-复杂背景\*-a.*')
    image_ORI_paths = glob.glob(r'D:\Data\Download\new_data\*-a.*')
    maxBorderLen = 800
    for imgoripath in image_ORI_paths:
        maskpath = get_path(imgoripath, 'mask')
        imgpixelpath = get_path(imgoripath, 'pixel')
        imgoutput_path = get_path(imgoripath, 'output')

        imgori = cv2_imread(imgoripath, toRGB=True, max_border_len=maxBorderLen)
        mask = cv2_imread(maskpath, shape=imgori.shape[0:2])[:,:,0] / 255.
        imgpixel = cv2_imread(imgpixelpath, toRGB=True, shape=imgori.shape[0:2])
        print(imgoripath, imgori.shape, mask.shape, imgpixel.shape)
        blue_bg = get_blue_bg(imgori)
        process(imgori, mask, blue_bg, imgpixel, imgoutput_path)
