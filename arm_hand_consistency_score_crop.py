import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from segmentation_mask_overlay.main import overlay_masks
import argparse
import os
import glob
import copy


parser = argparse.ArgumentParser( )
parser.add_argument('--arm_full_img', type=str, default=None,
                      help="path to left arm img")
parser.add_argument('--arm_full_pred', type=str, default=None,
                      help="path to left arm img")
parser.add_argument('--arm_half_img', type=str, default=None,
                      help="path to left arm img")
parser.add_argument('--arm_half_pred', type=str, default=None,
                      help="path to left arm img")
parser.add_argument('--arm_palm_img', type=str, default=None,
                      help="path to left arm img")
parser.add_argument('--arm_palm_pred', type=str, default=None,
                      help="path to left arm img")

parser.add_argument('--ori_fullarm', type=str, default=None,
                      help="path to left arm img")
parser.add_argument('--ori_palm', type=str, default=None,
                      help="path to left arm img")


parser.add_argument('--palm_img',type=str, default=None,
                      help="path to palm img ")
parser.add_argument('--best_prediction_folder', type=str, default=None,
                      help="path to best_prediction_folder ")
parser.add_argument('--path_gt', type=str, default=None,
                      help="path to gt annotations ")

parser.add_argument('--path_save_img', type=str, default=None,
                      help="path to save ")
parser.add_argument('--path_save_pred', type=str, default=None,
                      help="path to save ")
parser.add_argument('--path_save_gt', type=str, default=None,
                      help="path to save ")
parser.add_argument('--path_combine', type=str, default=None,
                      help="path to save ")

def is_image_file_png(filename):
  return any(filename.endswith(extension) for extension in [".png",".jpg"])


def convert_prediction(mask_prediction):
    out_pred = np.zeros_like(mask_prediction)
    R_pred = mask_prediction[:,:,0]
    G_pred = mask_prediction[:,:,1]
    B_pred = mask_prediction[:,:,2]
    R_out_pred = out_pred[:,:,0]
    G_out_pred = out_pred[:,:,1]
    B_out_pred = out_pred[:,:,2]
    pred_bg = (R_pred==0)&(G_pred==0)&(B_pred==0)
    pred_skin = (R_pred==255)&(G_pred==255)&(B_pred==255)
    pred_vitiligo = (R_pred==0)&(G_pred==0)&(B_pred==255)
    #same as gt
    R_out_pred[pred_skin] =255
    R_out_pred[pred_vitiligo]=255
    G_out_pred[pred_vitiligo] = 255
    B_out_pred[pred_vitiligo] = 255
    #cat
    out_pred = np.dstack((B_out_pred, G_out_pred, R_out_pred))
    return out_pred


def convert_gt(mask_gt):
    out_gt = np.zeros_like(mask_gt)
    R= copy.copy(out_gt)
    G = copy.copy(out_gt)
    B = copy.copy(out_gt)
    normal_skin_gt = (mask_gt>0.5)&(mask_gt<1.5)
    R[normal_skin_gt]=255.
    vitiligo_gt = (mask_gt>1.5)
    R[vitiligo_gt]=255.
    G[vitiligo_gt]=255
    B[vitiligo_gt]=255
    BG = mask_gt<0.5
    out_gt = np.dstack((B,G,R))
    return out_gt


def single_image_save(image,masks,masks_baseline,mask_gt,save_full_name):
    #convert mask into
    img = np.array(image)
    mask_pred = np.array(masks)
    mask_gt = np.array(mask_gt)
    #convert to ori shape
    out_pred = np.zeros_like(masks)
    R_pred = mask_pred[:,:,0]
    G_pred = mask_pred[:,:,1]
    B_pred = mask_pred[:,:,2]
    R_out_pred = out_pred[:,:,0]
    G_out_pred = out_pred[:,:,1]
    B_out_pred = out_pred[:,:,2]
    pred_bg = (R_pred==0)&(G_pred==0)&(B_pred==0)
    pred_skin = (R_pred==255)&(G_pred==255)&(B_pred==255)
    pred_vitiligo = (R_pred==0)&(G_pred==0)&(B_pred==255)
    #same as gt
    R_out_pred[pred_skin] =255
    R_out_pred[pred_vitiligo]=255
    G_out_pred[pred_vitiligo] = 255
    B_out_pred[pred_vitiligo] = 255
    #cat
    out_pred = np.dstack((B_out_pred, G_out_pred, R_out_pred))
    #convert mask into
    out_gt = np.zeros_like(masks)
    R= out_gt[:,:,0]
    G = out_gt[:,:,1]
    B = out_gt[:,:,2]
    normal_skin_gt = (mask_gt>0.5)&(mask_gt<1.5)
    R[normal_skin_gt]=255.
    vitiligo_gt = (mask_gt>1.5)
    R[vitiligo_gt]=255.
    G[vitiligo_gt]=255
    B[vitiligo_gt]=255
    BG = mask_gt<0.5
    out_gt = np.dstack((B,G,R))
    #BG, VITILIGO,NORMAL_SKIN
    output_masks=[]
    mask_labels=[]
    output_gt =[]
    output_masks.append(out_pred)
    output_gt.append(out_gt)
    mask_labels.append("pred_vitiligo")
    #output_masks.append(normal_skin)
    #mask_labels.append("pred_normal_skin")



    #convert mask into
    mask_baseline = np.array(masks_baseline)
    #BG, VITILIGO,NORMAL_SKIN

    output_masks.append(mask_baseline)
    mask_labels.append("gt_vitiligo")
    #output_masks.append(normal_skin_gt)
    #mask_labels.append("gt_normal_skin")



    # [Optional] prepare colors
    fig,axs = plt.subplots(1,4)
    axs[0].imshow(img)
    axs[0].set_title("original image")
    axs[1].imshow(out_gt)
    axs[1].set_title("groundtruth")
    axs[2].imshow(out_pred)
    axs[2].set_title("atrous5_2 result")
    axs[3].imshow(mask_baseline)
    axs[3].set_title("add_img result")
    fig.tight_layout()
    fig.savefig(save_full_name, bbox_inches="tight", dpi=300)


def save_arm_left_right_palm_img(arm_full_img_path, arm_half_img_path, arm_palm_img_path,palm_path, prediction_path, path_save_img,  path_ori_palm):
    names = [image_name for image_name in os.listdir(arm_half_img_path) if is_image_file_png(image_name)]
    for i in range(len(names)):
        arm_half_img_name = names[i]
        image_name = names[i].split('_')[0]
        patient_folder = names[i].split("-")[0]
        arm_full_img_list = list(glob.glob(os.path.join(arm_full_img_path,image_name+"*_contrast_score.png")))
        arm_palm_img_list = list(glob.glob(os.path.join(arm_palm_img_path, image_name+"*_contrast_score.png")))
        palm_img_list = list(glob.glob(os.path.join(path_ori_palm, patient_folder+"*")))
        if len(arm_full_img_list)==0:
            continue
        if len(palm_img_list)==0:
            continue
        if len(arm_palm_img_list)==0:
            continue
        assert len(arm_full_img_list)==1
        assert len(arm_palm_img_list)==1
        if i==87:
            print("pause here")
        assert len(palm_img_list)==1

        arm_half_img = os.path.join(arm_half_img_path, arm_half_img_name)
        arm_full_img = arm_full_img_list[0].split("/")[-1]
        arm_palm_img = arm_palm_img_list[0].split("/")[-1]
        palm_img = palm_img_list[0].split("/")[-1]


        arm_full_img_full_name = os.path.join(arm_full_img_path, arm_full_img)
        arm_palm_img_full_name = os.path.join(arm_palm_img_path, arm_palm_img)
        palm_img_full_name = os.path.join(palm_path, palm_img.replace('.jpg', '_contrast_score.png'))

        arm_full_img = Image.open(arm_full_img_full_name).convert("RGB")
        arm_half_img = Image.open(arm_half_img).convert("RGB")
        arm_palm_img = Image.open(arm_palm_img_full_name).convert("RGB")
        palm_img = Image.open(palm_img_full_name).convert("RGB")

        save_name = image_name + '.jpg'
        save_full_name = os.path.join(path_save_img, save_name)

        # [Optional] prepare colors
        print("printing image_{}".format(i))
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(arm_full_img)
        axs[0].set_axis_off()
        axs[1].imshow(arm_half_img)
        axs[1].set_axis_off()
        axs[2].imshow(arm_palm_img)
        axs[2].set_axis_off()
        axs[3].imshow(palm_img)
        axs[3].set_axis_off()
        fig.tight_layout()
        fig.savefig(save_full_name, bbox_inches="tight", dpi=300)


def gt_pre_name(img_name):
    pred = img_name.replace('_contrast_score.png','_mask.png')
    gt_name = img_name.replace('_contrast_score.png','.png')
    return pred, gt_name

def palm_gt_pre_name(img_name):
    pred = img_name.replace('.jpg','_mask.png')
    gt_name = img_name.replace('.jpg','.png')
    return pred, gt_name


def save_arm_left_right_palm_ann(arm_full_img_path, arm_half_img_path, arm_palm_img_path,palm_path, prediction_path, path_save_img, path_ori_palm,path_gt,path_save_pred,path_save_gt):

    names = [image_name for image_name in os.listdir(arm_half_img_path) if is_image_file_png(image_name)]
    for i in range(len(names)):
        arm_half_img_name = names[i]
        armhalf_pred_name,armhalf_gt_name = gt_pre_name(arm_half_img_name)
        half_pred_path = arm_half_img_path.rsplit('/', 2)[0]
        armhalf_pred = os.path.join(half_pred_path,'output',armhalf_pred_name)
        arm_gt = os.path.join(path_gt,arm_half_img_name.replace('_contrast_score.png','.png'))

        patient_folder = names[i].split("-")[0]
        image_name = names[i].split('_')[0]
        arm_full_img_list = list(glob.glob(os.path.join(arm_full_img_path,image_name+"*_contrast_score.png")))
        arm_palm_img_list = list(glob.glob(os.path.join(arm_palm_img_path, image_name+"*_contrast_score.png")))
        palm_img_list = list(glob.glob(os.path.join(path_ori_palm, patient_folder + "*")))

        if len(arm_full_img_list)==0:
            continue
        if len(palm_img_list)==0:
            continue
        if len(arm_palm_img_list)==0:
            continue
        assert len(arm_full_img_list)==1
        assert len(arm_palm_img_list)==1
        assert len(palm_img_list)==1

        arm_full_img = arm_full_img_list[0].split("/")[-1]
        armfull_pred_name, _ = gt_pre_name(arm_full_img)
        full_pred_path = arm_full_img_path.rsplit('/',2)[0]
        armfull_pred = os.path.join(full_pred_path,'output', armfull_pred_name)


        arm_palm_img = arm_palm_img_list[0].split("/")[-1]
        armpalm_pred_name, _ = gt_pre_name(arm_palm_img)
        half_pred_path = arm_palm_img_path.rsplit('/', 2)[0]
        armpalm_pred = os.path.join(half_pred_path,'output', armpalm_pred_name)

        palm_img = palm_img_list[0].split("/")[-1]
        palm_pred_name, palm_gt_name = palm_gt_pre_name(palm_img)
        half_pred_path = palm_path.rsplit('/', 2)[0]
        palm_pred = os.path.join(half_pred_path,'output', palm_pred_name)
        palm_gt_full_name = os.path.join(path_gt,palm_gt_name)



        arm_half_pred = cv2.imread(armhalf_pred)
        arm_half_pred = convert_prediction(arm_half_pred)
        arm_half_gt = cv2.imread(arm_gt,cv2.IMREAD_GRAYSCALE)
        arm_half_gt = convert_gt(arm_half_gt)

        arm_full_pred = cv2.imread(armfull_pred)
        arm_full_pred =convert_prediction(arm_full_pred)
        arm_full_gt = arm_half_gt

        arm_palm_pred = cv2.imread(armpalm_pred)
        arm_palm_pred = convert_prediction(arm_palm_pred)
        arm_palm_gt = arm_half_gt

        palm_pred = cv2.imread(palm_pred)
        palm_gt = cv2.imread(palm_gt_full_name,cv2.IMREAD_GRAYSCALE)
        palm_pred = convert_prediction(palm_pred)
        palm_gt = convert_gt(palm_gt)


        save_name = image_name + '.jpg'
        save_pred_name = os.path.join(path_save_pred,save_name)
        save_full_name_gt = os.path.join(path_save_gt, save_name)



        # [Optional] prepare colors
        print("printing predictions_{}".format(i))
        fig, axs = plt.subplots(1,4)
        axs[0].imshow(arm_full_pred)
        axs[0].set_axis_off()
        axs[1].imshow(arm_half_pred)
        axs[1].set_axis_off()
        axs[2].imshow(arm_palm_pred)
        axs[2].set_axis_off()
        axs[3].imshow(palm_pred)
        axs[3].set_axis_off()
        fig.tight_layout()
        fig.savefig(save_pred_name, bbox_inches="tight", dpi=300)
        plt.close('all')

        # [Optional] prepare colors
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(arm_full_gt)
        axs[0].set_axis_off()
        axs[1].imshow(arm_half_gt)
        axs[1].set_axis_off()
        axs[2].imshow(arm_palm_gt)
        axs[2].set_axis_off()
        axs[3].imshow(palm_gt)
        axs[3].set_axis_off()
        fig.tight_layout()
        fig.savefig(save_full_name_gt, bbox_inches="tight", dpi=300)
        plt.close('all')


def combined_img(path_save_img, path_pred, path_save_gt, path_save_combined):
    names = [image_name for image_name in os.listdir(path_save_img) if is_image_file_png(image_name)]
    for i in range(len(names)):
        img_name = names[i]
        img_full_name = os.path.join(path_save_img,img_name)
        pred_full_name = os.path.join(path_pred, img_name)
        gt_full_name = os.path.join(path_save_gt,img_name)
        save_full_name = os.path.join(path_save_combined,img_name)

        img = Image.open(img_full_name).convert("RGB")
        pred = Image.open(pred_full_name).convert("RGB")
        gt = Image.open(gt_full_name).convert("RGB")
        print("printing combined_{}".format(i))
        fig, axs = plt.subplots(3, 1)
        axs[0].imshow(img)
        axs[0].set_title("original image")
        axs[0].set_axis_off()
        axs[1].imshow(gt)
        axs[1].set_title("gt")
        axs[1].set_axis_off()
        axs[2].imshow(pred)
        axs[2].set_title("pred")
        axs[2].set_axis_off()

        fig.tight_layout()
        fig.savefig(save_full_name, bbox_inches="tight", dpi=300)





if __name__ == '__main__':
  args = parser.parse_args()
  arm_full_img_path = args.arm_full_img
  arm_full_pred_path = args.arm_full_pred

  arm_half_img_path = args.arm_half_img
  arm_half_pred_path = args.arm_half_pred

  arm_palm_img_path = args.arm_palm_img
  arm_palm_pred_path = args.arm_palm_pred


  palm_path  = args.palm_img
  prediction_path = args.best_prediction_folder
  path_gt = args.path_gt

  path_save_img = args.path_save_img
  path_save_pred= args.path_save_pred
  path_save_gt = args.path_save_gt
  path_save_combined = args.path_combine
  path_ori_palm = args.ori_palm


  save_arm_left_right_palm_img(arm_full_img_path, arm_half_img_path, arm_palm_img_path,palm_path, prediction_path, path_save_img, path_ori_palm)
  save_arm_left_right_palm_ann(arm_full_img_path, arm_half_img_path, arm_palm_img_path,palm_path, prediction_path, path_save_img, path_ori_palm,path_gt,path_save_pred,
                               path_save_gt)
  combined_img(path_save_img, path_save_pred, path_save_gt, path_save_combined)

  # vitiligo path, skin_bg_path: RGB image of prediction


