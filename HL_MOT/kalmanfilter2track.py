# Import python libraries
import numpy as np
from filterpy.kalman import KalmanFilter
import math

class KalmanFilterObj(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0

  def __init__(self, input, frame_num, dim_x, dim_z):
    """
    Initialises a tracker
    """
    # define constant velocity model
    # self.kf = KalmanFilter(dim_x, dim_z)
    # self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])

    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])

    # self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
    #                       [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
    # self.kf.F = np.zeros((dim_x, dim_x), int)
    # np.fill_diagonal(self.kf.F, 1)
    # print(self.kf.F)

    self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
    # self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
    # self.kf.H = np.zeros((dim_z, dim_z), int)
    # np.fill_diagonal(self.kf.H, 1)
    # temp = np.zeros((dim_z, (dim_x - dim_z)), int)
    # self.kf.H = np.concatenate((self.kf.H, temp), axis=1)
    # print(self.kf.H)
    # print(self.kf.H.shape)
    self.kf.R[2:, 2:] *= 10.
    # self.kf.R[2:,2:] *= 10.
    # print(self.kf.R.shape)
    # print(self.kf.R)
    self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
    # self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    # print(self.kf.P.shape)
    # print(self.kf.P)
    self.kf.P *= 10.
    # self.kf.P *= 10.
    # print(self.kf.P.shape)
    # print(self.kf.P)
    self.kf.Q[-1, -1] *= 0.01
    # self.kf.Q[-1,-1] *= 0.01
    # print(self.kf.Q.shape)
    # print(self.kf.Q)
    self.kf.Q[4:, 4:] *= 0.01
    # self.kf.Q[4:,4:] *= 0.01
    # print(self.kf.Q.shape)
    # print(self.kf.Q)

    # print(self.kf.x)
    # print (state_input)

    # predict the bounding box coordinate
    # state_input = input[0]
    # self.kf.x[0] = state_input[0]
    # self.kf.x[1] = state_input[1]
    # self.kf.x[2] = state_input[2]
    # self.kf.x[3] = state_input[3]

    self.kf.x[:4] = self.convert_bbox_to_z(input[0])
    # if ((self.kf.x[6] + self.kf.x[2]) <= 0):
    #   self.kf.x[6] *= 0.0

    # predict centre point of object bounding box
    # state_input = input[1]
    # self.kf.x[0] = state_input[0]
    # self.kf.x[1] = state_input[1]

    # print (input)
    # print (input[0])
    # self.kf.x[:4] = np.reshape(state_input, (4, 1))
    # print(self.kf.x)
    # self.kf.predict()
    # advance the state

    self.input = input
    self.time_since_update = 0
    self.id = KalmanFilterObj.count
    KalmanFilterObj.count += 1
    self.history = []
    self.prediction_history = []
    # self.currentrecord = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    # self.record = []
    # self.type = self.input[3]
    # self.first_pos = input [0]
    # self.current_trj = input[0]  # the last one is the feature index

    # self.frame_num = frame_num
    # self.currentrecord.append(input [0])

    # self.currentstat = True #True for detection and false for prediction
    self.pred_conf = False # overlap status between prediction and detection, by default set as False
    # self.frame_start = frame_num
    # self.frame_end = frame_num
    # self.frame_detection= []
    # self.frame_prediction = []

    self.time_stamp = []
    # self.frame_firstseen =
    # self.frame_lastseen =
    # print ("currnet record")
    # print (self.currentrecord)
  # def convert_state_to_center(self):
  #     """
  #     Takes a state and returns its center point
  #     """
  #     # output = np.array([(input[0]+input[1]) / 2, (input[2]+input[3])/2]).reshape((1, 2))
  #     output = np.array([(self.kf.x[0] + self.kf.x[1]) / 2, (self.kf.x[2] + self.kf.x[3]) / 2]).reshape((1, 2))
  #     # print (output)
  #     return output

  def remove_trk_final_predictions (self, last_k_time):
    # Remove the consecutive predicted trajectories of an terminating track
    ori_len = len(self.history)
    i = len(self.history)
    for trk in reversed(self.history):
      i -= 1
      if (i >= ori_len - last_k_time):
        self.history.pop(i)
        self.time_stamp.pop(i)

  # def interpolate_history (self):
  #   for i in range(len(self.time_stamp)):
  #     if (self.time_stamp[i][1]=='BP'):
  #       if ():
  #         #manual interpolate with respect to previous position
  #       else:
  #         #just back the previous reliable position (GP or D)
  #     elif ():


    # Remove the consecutive predicted trajectories of an terminating track
    # ori_len = len(self.history)
    # i = len(self.history)
    # for trk in reversed(self.history):
    #   i -= 1
    #   if (i >= ori_len - last_k_time):
    #     self.history.pop(i)
    #     self.time_stamp.pop(i)


  def update_unmatched(self, overlap_thres, current_frame):
    """
    Updates the state vector with observed bbox, only called for unmatched tracks
    """
    for idx, val in self.reverse_enum(self.time_stamp):
      # if (val[1] == "D"):
      if (val[1] == "D" or val[1] == "GP"):
        # if (self.compute_IOU(self.convert_z_to_bbox(self.kf.x).tolist(), self.retrieve_hist(idx)) >= 0.5):
        if (self.compute_IOU(self.convert_z_to_bbox(self.kf.x).tolist(), self.retrieve_hist(idx)) < overlap_thres):
          pred_stat = "BP"
          # remove the bad prediction results and replace with last seen detection of the obj
          self.history.pop()
          self.history.append(self.retrieve_hist(idx))
        else:
          pred_stat = "GP"
        break
      else:
        #no reference detection to replace for unreliable prediction
        pred_stat = "UP"

    self.time_stamp.pop()
    self.time_stamp.append([current_frame, pred_stat])

  def reverse_enum(self, L):
    for index in reversed(range(len(L))):
      yield index, L[index]

  def update_matched(self, detect_in, detect_index, current_frame):
    """
    Updates the state vector with observed bbox, only called for matched tracks
    """
    self.time_since_update = 0
    self.prediction_history = []

    # remove the prediction results and replace with detection result start
    self.history.pop()
    self.history.append(detect_in[detect_index][0])


    # self.frame_prediction.pop()

    self.time_stamp.pop()
    self.time_stamp.append([current_frame, "D"])
    # remove the prediction results and replace with detection result end

    # self.frame_detection.append(current_frame)
    # self.current_trj = detect_in[detect_index][1]
    self.hits += 1
    self.hit_streak += 1
    # self.history.append(detect_in[detect_index][0])
    self.kf.update(self.convert_bbox_to_z(detect_in[detect_index][0]))

    self.kf.x[:4] = self.convert_bbox_to_z(detect_in[detect_index][0]) # replace detection directly if it is matched
    # self.currentstat = True
    self.input = detect_in[detect_index]
    # print ("checking the input")
    # print (self.input)
    # self.type = detect_in[detect_index][3]

    # self.current_trj = detect_in[detect_index][0] #the last one is the feature index
    # self.currentrecord.append(detect_in[detect_index][0])
    # self.goodpredict = False

    # self.frame_end = current_frame
    # print ("currnet record")
    # print (self.currentrecord)

  def compute_IOU(self, F1, F2):
    # Reading the list item in Frame [ymin, ymax, xmin, xmax]
    width1 = F1[3] -F1[2]
    height1 = F1[1] - F1[0]

    width2 = F2[3] - F2[2]
    height2 = F2[1] - F2[0]

    start_x = min (F1[2], F2[2])
    end_x = max ((F1[2]+width1), (F2[2]+width2))
    width = width1 + width2 - (end_x-start_x)

    start_y = min (F1[0], F2[0])
    end_y = max ((F1[0]+height1), (F2[0]+height2))
    height = height1 + height2 - (end_y - start_y)

    if ((width <=0) or (height <= 0)):
      result = 0
    else:
      result = (height*width)/ float((height1*width1)+(height2*width2)-(height*width))
    return result

  def predict_track (self, current_frame):
      # predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0

    self.kf.predict()
    # self.pred_conf = True
    # self.currentstat = False
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0

    self.history.append(self.convert_z_to_bbox(self.kf.x).tolist())
    self.prediction_history.append(self.convert_z_to_bbox(self.kf.x).tolist())

    # self.frame_prediction.append(current_frame)
    # print ("what is it?")
    # print (self.convert_z_to_bbox(self.kf.x).tolist())
    # print (detect_in[detect_index][0])

    self.time_stamp.append([current_frame, "P"])

    # print (len(self.history))
      # self.hits = 0

    self.time_since_update += 1

    # self.frame_end = current_frame
    # self.history.append(self.convert_z_to_bbox(self.kf.x).tolist())

    # self.currentrecord.append(self.history[-1])
    # print ("currnet record")
    # print (self.currentrecord)
    return self.history[-1]
    # return self.history

  # def save_pred_conf(self, stat):
  #   self.pred_conf = stat
  #
  # def retrieve_pred_conf (self):
  #   return self.pred_conf

  def retrieve_time_stamp (self, iter):
    # return self.time_stamp[iter-1]
    return self.time_stamp[iter]

  def retrieve_time_stamp_all (self):
    return self.time_stamp

  def retrieve_obj_info(self):
    return  self.input

  # def retrieve_trj_info(self):
  #   retrieve_output = 0
  #   idx = 0
  #   for idx, val in self.reverse_enum(self.time_stamp):
  #     if (val[1] == "D"):
  #       retrieve_output = self.retrieve_hist(idx)
  #       break
  #   # return  output
  #   #
  #   return [retrieve_output, idx]

  def retrieve_id(self):
    return  self.id

  def retrieve_type(self):
    return  self.input[3]

  def retrieve_hist_whole(self):
    return  self.history

  def retrieve_hist(self, iter):
    return self.history[iter]

  # def retrieve_hist_all(self):
  #   return self.history

  def retrieve_time_stamp_sel(self, num):
    return self.time_stamp[:-(num+1):-1]

  # def retrieve_predicthist(self, iter):
  #   return  self.prediction_history[iter-1]

  def retrieve_frame_start(self):
    # out = self.retrieve_time_stamp_all()[0]
    # return  self.frame_start
    return self.time_stamp[0]

  def retrieve_frame_end(self):
    sz = len (self.time_stamp)
    return self.time_stamp[sz-1]
    # return  self.frame_end

  # def add_record(self, input, frame_num):
  #   self.record.append ([input, frame_num])
  #   return self.record [-1]

  def retrieve_state(self):
    """
    Returns the current bounding box estimate.
    """
    # return KalmanFilterObj.convert_state_to_center(self.kf.x)
    # return self.kf.x

    return self.convert_z_to_bbox(self.kf.x)

  # def retrieve_record(self):
  #   """
  #   Returns the current bounding box estimate.
  #   """
  #   # return KalmanFilterObj.convert_state_to_center(self.kf.x)
  #   # return self.kf.x
  #
  #   return self.currentrecord[-1]

  def convert_state_to_center(self, x):
    return (np.concatenate((((x[0] + x[1])/2), ((x[2] + x[3])/2)), axis=0))

      # print (np.array(((x[0] + x[1]) / 2), ((x[3] + x[4]) / 2)))
      # return np.array(((x[0] + x[1]) / 2), ((x[3] + x[4]) / 2))
        #
        # """
        # Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
        #   [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
        # """
    # print (np.array(((x[0] + x[1])/2),((x[3] + x[4])/2)))
    # return np.array(((x[0] + x[1])/2),((x[3] + x[4])/2))

  # def convert_state_to_bb(self, x):
  #     concat = np.concatenate((x[0], x[1], x[2], x[3]), axis=0)
  #     # print (concat)
  #     # concat = np.concatenate(concat, x[2], axis=0)
  #     # concat = np.concatenate(concat, x[3], axis=0)
  #     return (concat)

  def convert_bbox_to_z(self,bbox):
    """
    Takes a bounding box in the form [ymin, ymax, xmin, xmax] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    # print("check convert")
    # print (bbox)
    w = bbox[3] - bbox[2]
    h = bbox[1] - bbox[0]
    x = (bbox[3] + bbox[2]) / 2.
    y = (bbox[1] + bbox[0]) / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    # print (np.array([x, y, s, r]).reshape((4, 1)))
    return np.array([x, y, s, r]).reshape((4, 1))

  def convert_z_to_bbox(self, z):
      """
      Takes z in the form of [x, y, s, r] and convert to bounding box in the form [ymin, ymax, xmin, xmax]
      """
      w = math.sqrt(z[2] * z[3])
      h = w / float(z[3])

      x_max = 0.5 * (2*z[0] + w)
      x_min = x_max - w
      y_max = 0.5 * (2*z[1] + h)
      y_min = y_max - h

      # print (np.array([y_min, y_max, x_min, x_max]).reshape((4, 1)))
      # print (type (np.array([y_min, y_max, x_min, x_max]).reshape((4, 1))))
      return np.concatenate((y_min, y_max, x_min, x_max), axis=0)






