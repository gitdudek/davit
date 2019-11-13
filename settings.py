'''IOU Tracker settings'''
if "DPM" in seq:
    sigma_l = -0.5
    sigma_h = 0.5
    sigma_iou = 0.5
    t_min = 4
elif "FRCNN" in seq:
    sigma_l = 0.0
    sigma_h = 0.9
    sigma_iou = 0.4
    t_min = 3
elif "SDP" in seq:
    sigma_l = 0.4
    sigma_h = 0.5
    sigma_iou = 0.3
    t_min = 2
else:
    sigma_l = 0.25       
    sigma_h = 0.45      
    sigma_iou = 0.2    
    t_min = 3

'''DAVIT-Algorithm settings for MOT17 evaluation '''
method = ['TemplateMatching','KCF']
v_ttl = [2,4,6,8,10,12,14,16,18,20] #list of lenghts for visual object tracking
sigma_iou_ttl = [0,0.2,0.4,0.6,0.8,1] #list of IOU thresholds for merging procedure