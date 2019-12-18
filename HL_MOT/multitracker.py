# Import python libraries
import numpy as np
from kalmanfilter2track import KalmanFilterObj
from filterpy.kalman import KalmanFilter
# from common import dprint
from scipy.optimize import linear_sum_assignment
import cv2
from scipy.optimize import fmin
import scipy

class Tracker(object):
    """Tracker class that updates track vectors of object tracked
    Attributes:
        None
    """
    def __init__(self, cost_feat_thresh, max_trace_length, track_step_overlap, min_hits, dist_thresh, im_height, im_width):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: thrsh for distance comparison
            cost_feat_thresh
            distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_length: maximum trace path history length
            min_hits: minimum number of detection to disregard false detection
            trackCount: count of each track object
        Return:
            None
        """

        self.cost_feat_thresh = cost_feat_thresh
        self.dist_thresh = dist_thresh
        # self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.min_hits = min_hits
        self.trackstepoverlap = track_step_overlap
        # self.hits = 0
        self.tracks_per_frame = []
        self.alltracks = []
        self.active_tracks = []
        self.im_height = im_height
        self.im_width = im_width
        self.trackCount = 0
        self.framecount = 0
        # self.operation_mode = ope_mode

    def FeatureExtractorbyFeat(self, list_in, feat_index):
        # Extract and Reshape list[feat][obj] according to called feature index
        list_ori = list_in[feat_index]
        list_selected = []
        for i in range(len(list_ori)):
            list_selected.append(np.reshape((np.asarray(list_ori[i])), (np.asarray(list_ori[i]).shape[0], 1)))
        return  list_selected


    def FeatureExtractorbyObj(self, list_in, feat_index):
        # Extract and Reshape list[obj][feat] according to called feature index
        n = len (list_in)
        list_selected = []
        for i in range(len(list_in)):
            list_selected.append (list_in[i][feat_index])
        return  list_selected

    def ListConversion(self, list_in):
        # Convert list_a[feat][obj] with length of n features into list_b[obj][feat] with length of k objects
        # n_obj = len(list_in[0])
        list_output=[]
        for x in range (len(list_in[0])):
            list_temp = []
            for y in range(len(list_in)):
                list_temp.append(list_in[y][x])
            list_output.append(list_temp)
        return  list_output

    def FlattenAllFeat(self, list_in):
        # Flatten for processing in kalman filter
        list_output = []
        for i in range(len(list_in)):
            for j in range (len(list_in[i])):
                list_output.append(list_in[i][j])
        return  list_output

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

    # def retrieve_trk_obj(self):
    #     return self.alltracks.retrieve_obj_info()

    def UpdateTracker(self, detected_per_frame, feat_index, frame_num):
        feat_pos_iou = False
        feat_pos_c = False
        feat_pos_bb = False
        feat_clr = False
        feat_lbl = False
        feat_reid = False

        self.active_tracks = []
        self.framecount = frame_num
        # detections = self.ListConversion(detected_per_frame)  #detections[obj][feat]

        # HL 2019 change the way feature are fed into the tracker
        detections = detected_per_frame
        if 0 in feat_index:
            feat_pos_iou = True
        if 1 in feat_index:
            feat_pos_bb = True
        if 2 in feat_index:
            feat_pos_c = True
        if 3 in feat_index:
            feat_clr = True
        if 4 in feat_index:
            feat_lbl = True
        if 5 in feat_index:
            feat_reid = True

        detection_frame_pos_iou = []
        detection_frame_pos_c = []
        detection_frame_pos_bb = []
        detection_frame_clr = []
        detection_frame_lbl = []
        detection_frame_reid = []

        tracked_frame_pos_iou = []
        tracked_frame_pos_c = []
        tracked_frame_pos_bb = []
        tracked_frame_clr = []
        tracked_frame_lbl = []
        tracked_frame_reid = []

        # Create tracks if no tracks vector found
        if (len(self.tracks_per_frame) == 0):  # might wanna add conditions here later, at some point where all objects left scene?
            # print ("no tracks found, assign all detection to tracks")
            # HL: Should to initialize individual track property and history here
            for t, trk in enumerate(detections):
                # obj = KalmanFilterObj(detections[t], dim_x=6, dim_z=4)
                obj = KalmanFilterObj(detections[t], self.framecount, dim_x=7, dim_z=4)
                #  dim_x = for all the information contained by the object
                #  dim_z = for the bounding box coordinate prediction
                self.tracks_per_frame.append(obj)
                self.trackCount += 1
        else:
            print ("Here is length for existing frame track")
            print (len(self.tracks_per_frame))


            for t, trk in enumerate(self.tracks_per_frame):
                track_feat = trk.retrieve_obj_info()

                #if I wanna turn off prediction
                # predict_box = trk.predict_track(frame_num)
                # detections = self.ListConversion(detected_per_frame)

                predict_box = trk.predict_track(frame_num)

                if (feat_pos_iou):
                    tracked_frame_pos_iou.append (track_feat[0])
                if (feat_pos_bb):
                    tracked_frame_pos_bb.append (track_feat[0])
                if (feat_pos_c):
                    tracked_frame_pos_c.append(track_feat[1])
                if (feat_clr):
                    tracked_frame_clr.append (track_feat[2])
                if (feat_lbl):
                    tracked_frame_lbl.append(track_feat[3])
                if (feat_reid):
                    tracked_frame_reid.append(track_feat[4])

            for t, trk in enumerate(detections):
                if (feat_pos_iou):
                    detection_frame_pos_iou.append(self.FeatureExtractorbyObj(detections, 0)[t])
                if (feat_pos_bb):
                    detection_frame_pos_bb.append(self.FeatureExtractorbyObj(detections, 0)[t])
                if (feat_pos_c):
                    detection_frame_pos_c.append (self.FeatureExtractorbyObj(detections, 1)[t])
                if (feat_clr):
                    detection_frame_clr.append(self.FeatureExtractorbyObj(detections, 2)[t])
                if (feat_lbl):
                    detection_frame_lbl.append(self.FeatureExtractorbyObj(detections, 3)[t])
                if (feat_reid):
                    detection_frame_reid.append(self.FeatureExtractorbyObj(detections, 4)[t])

            # Calculate cost using sum of square distance between tracks and detection
            # Note that this particular cost is computed for box coordinate feature
            if (feat_pos_iou):
                cost_pos_iou = np.zeros(shape=(len(tracked_frame_pos_iou), len(detection_frame_pos_iou)))  # Cost matrix
                diff_pos_iou = np.zeros(shape=(len(tracked_frame_pos_iou), len(detection_frame_pos_iou)))

                for i in range(len(tracked_frame_pos_iou)):
                    for j in range(len(detection_frame_pos_iou)):
                        cost_pos_iou[i, j] = 1 - self.compute_IOU(tracked_frame_pos_iou[i], detection_frame_pos_iou[j])  # low cost indicates high IOU

            ########################################################################################
            if (feat_pos_bb):
                cost_pos_bb = np.zeros(shape=(len(tracked_frame_pos_bb), len(detection_frame_pos_bb)))  # Cost matrix
                diff_pos_bb = np.zeros(shape=(len(tracked_frame_pos_bb), len(detection_frame_pos_bb)))

                # for i in range(len(tracked_frame_pos_bb)):
                #     norm_track_bb = [tracked_frame_pos_bb[i][0]/self.im_height, tracked_frame_pos_bb[i][1]/self.im_height, tracked_frame_pos_bb[i][2]/self.im_width, tracked_frame_pos_bb[i][3]/self.im_width]
                #     for j in range(len(detection_frame_pos_bb)):
                #         norm_detect_bb = [detection_frame_pos_bb[j][0] / self.im_height,
                #                        detection_frame_pos_bb[j][1] / self.im_height,
                #                        detection_frame_pos_bb[j][2] / self.im_width,
                #                        detection_frame_pos_bb[j][3] / self.im_width]
                #         cost_pos_bb[i][j] = np.sqrt(
                #             ((norm_track_bb[0] - norm_detect_bb[0]) ** 2) + ((norm_track_bb[1] - norm_detect_bb[1]) ** 2) + ((norm_track_bb[2] - norm_detect_bb[2]) ** 2) + ((norm_track_bb[3] - norm_detect_bb[3]) ** 2))

                # Calculating the average distance of corners of bb, then compare it with fixed thres as distance similarity
                for i in range(len(tracked_frame_pos_bb)):
                    for j in range(len(detection_frame_pos_bb)):
                        # cost_pos_bb[i][j] = 0.25 * ((tracked_frame_pos_c[i] [0]- detection_frame_pos_c[j][0]) + (tracked_frame_pos_c[i] [1]- detection_frame_pos_c[j][1]) + (tracked_frame_pos_c[i] [2]- detection_frame_pos_c[j][2]) + (tracked_frame_pos_c[i] [3]- detection_frame_pos_c[j][3]))
                        # cost_pos_bb[i][j] = 1- (max (0,((0.25 * ((tracked_frame_pos_bb[i] [0]- detection_frame_pos_bb[j][0]) + (tracked_frame_pos_bb[i] [1]- detection_frame_pos_bb[j][1]) + (tracked_frame_pos_bb[i] [2]- detection_frame_pos_bb[j][2]) + (tracked_frame_pos_bb[i] [3]- detection_frame_pos_bb[j][3])))-self.dist_thresh)/self.dist_thresh))
                        # cost_pos_bb[i][j] = 1 - (max(0, ((0.25 * (abs(tracked_frame_pos_bb[i][0] - detection_frame_pos_bb[j][0]) + abs(tracked_frame_pos_bb[i][1] - detection_frame_pos_bb[j][1]) + abs(tracked_frame_pos_bb[i][2] - detection_frame_pos_bb[j][2]) + abs(tracked_frame_pos_bb[i][3] - detection_frame_pos_bb[j][3]))) - self.dist_thresh) / self.dist_thresh))
                        cost_pos_bb[i][j] = (1-max(0,((self.dist_thresh-(((0.25*(abs(tracked_frame_pos_bb[i][0]- detection_frame_pos_bb[j][0])+abs(tracked_frame_pos_bb[i][1]- detection_frame_pos_bb[j][1])+abs(tracked_frame_pos_bb[i][2]- detection_frame_pos_bb[j][2])+abs(tracked_frame_pos_bb[i][3]- detection_frame_pos_bb[j][3]))))))/self.dist_thresh)))
                # print ("cost pos")
                # print (cost_pos_bb)

            ########################################################################################
            if (feat_pos_c):
                cost_pos_c = np.zeros(shape=(len(tracked_frame_pos_c), len(detection_frame_pos_c)))  # Cost matrix
                diff_pos_c = np.zeros(shape=(len(tracked_frame_pos_c), len(detection_frame_pos_c)))

                # for i in range(len(tracked_frame_pos_c)):
                    # norm_track_c = [tracked_frame_pos_c[i][0] / self.im_width,
                    #                   tracked_frame_pos_c[i][1] / self.im_height]
                        # norm_track = [tracked_frame_pos_c[i][0]/self.im_height, tracked_frame_pos_c[i][1]/self.im_height, tracked_frame_pos_c[i][2]/self.im_width, tracked_frame_pos_c[i][3]/self.im_width]

                    # for j in range(len(detection_frame_pos_c)):
                            # diff_pos_c = tracked_frame_pos_c[i] - detection_frame_pos[j]
                        # norm_detect_c = [detection_frame_pos_c[j][0] / self.im_width,
                        #                    detection_frame_pos_c[j][1] / self.im_height]
                        # cost_pos_c[i][j] = np.sqrt(((norm_track_c[0]-norm_detect_c[0])**2)+((norm_track_c[1]-norm_detect_c[1])**2))


            ########################################################################################
            if (feat_lbl):
                cost_lbl = np.zeros(shape=(len(tracked_frame_lbl), len(detection_frame_lbl)))  # Cost matrix
                diff_lbl = np.zeros(shape=(len(tracked_frame_lbl), len(detection_frame_lbl)))
                for i in range(len(tracked_frame_lbl)):
                    for j in range(len(detection_frame_lbl)):
                        # diff_lbl = tracked_frame_lbl[i] - detection_frame_lbl[j]
                        # distance_lbl = np.sqrt(diff_lbl[0][0] * diff_lbl[0][0] + diff_lbl[1][0] * diff_lbl[1][0])
                        # cost_lbl[i][j] = (0.5) * distance_lbl
                        if (tracked_frame_lbl[i][0]==detection_frame_lbl[j][0]):
                            # cost_lbl[i, j] = 1- ((tracked_frame_lbl[i][1])*(detection_frame_lbl[j][1]))
                            cost_lbl[i, j] = 1 - 0.5* (((tracked_frame_lbl[i][1])+(detection_frame_lbl[j][1])))
                        else:
                            cost_lbl[i, j] = 1

            ########################################################################################

            if (feat_clr):
                cost_clr = np.zeros(shape=(len(tracked_frame_clr), len(detection_frame_clr)))  # Cost matrix
                for i in range(len(tracked_frame_clr)):
                    for j in range(len(detection_frame_clr)):
                        cost_clr[i][j] = cv2.compareHist(tracked_frame_clr[i], detection_frame_clr[j], cv2.HISTCMP_BHATTACHARYYA)

                # row_ind_clr, col_ind_clr = linear_sum_assignment(cost_clr)
                # matched_indices_clr = np.concatenate((row_ind_clr.reshape((row_ind_clr.shape[0], 1)), col_ind_clr.reshape((col_ind_clr.shape[0], 1))),axis=1)
                # print (row_ind_clr)
                # print (col_ind_clr)
                # print (matched_indices_clr)

            ########################################################################################
            if (feat_reid):
                cost_reid = np.zeros(shape=(len(tracked_frame_reid), len(detection_frame_reid)))  # Cost matrix
                diff_reid = np.zeros(shape=(len(tracked_frame_reid), len(detection_frame_reid)))
                for i in range(len(tracked_frame_reid)):
                    for j in range(len(detection_frame_reid)):
                        # print (tracked_frame_reid[i].shape)
                        # print (detection_frame_reid[j].shape)
                        # dist = scipy.spatial.distance.euclidean(tracked_frame_reid[i], detection_frame_reid[j])
                        cost_reid[i][j] = 1 - ((scipy.spatial.distance.euclidean(tracked_frame_reid[i], detection_frame_reid[j]))/100)

                        # input("Press Enter to continue...")
                        # tracked_frame_clr[i]
                        # detection_frame_clr[j]

            ########################################################################################
            # print (cost_reid)
            # print (type(cost_reid))
            # print (cost_reid.shape)
            # map(max, cost_reid)
            # map(min, cost_reid)
            # print (max(cost_reid.flatten()))
            # print(min(cost_reid.flatten()))
            # input("Press Enter to continue...")

            if (feat_pos_bb and feat_clr and feat_lbl and (not feat_reid)):
                # cost_combined = (cost_pos_bb + cost_lbl + cost_clr) / 3
                # cost_combined = 0.8 * (cost_pos_bb) + 0.1 * (cost_clr) + 0.1 *(cost_lbl)
                cost_combined = 0.6 * (cost_pos_bb) + 0.3 * (cost_clr) + 0.1 * (cost_lbl) #ICIAR feature weight
            elif (feat_pos_bb and feat_clr and feat_lbl and feat_reid):
                cost_combined = (cost_pos_bb + cost_lbl + cost_clr + cost_reid) / 4
            elif(feat_pos_iou and feat_clr and feat_lbl):
                cost_combined = (cost_pos_iou + cost_lbl + cost_clr) / 3
            elif (feat_pos_iou and feat_clr and feat_lbl and feat_pos_bb):
                cost_combined = (cost_pos_iou + cost_pos_bb + cost_lbl + cost_clr) / 4
            else:
                # print ("sss")
                # cost_combined = (cost_lbl + cost_clr) / 2
                # cost_combined = cost_clr
                cost_combined = cost_pos_bb
                # cost_combined = cost_lbl
                # cost_combined = cost_reid

            row_ind, col_ind = linear_sum_assignment(cost_combined)
            matched_indices = np.concatenate((row_ind.reshape((row_ind.shape[0], 1)), col_ind.reshape((col_ind.shape[0], 1))), axis=1)
            # print ("assign mat")
            # print (row_ind)
            # print (col_ind)
            # print (matched_indices)

            unmatched_detections = []
            for d, det in enumerate(detections):
                if (d not in matched_indices[:, 1]):
                    unmatched_detections.append(d)
            unmatched_trackers = []
            for t, trk in enumerate(self.tracks_per_frame):
                if (t not in matched_indices[:, 0]):
                    unmatched_trackers.append(t)

            # remove matches with too high cost
            matches = []
            for m in matched_indices:
                if (cost_combined[m[0], m[1]] < self.cost_feat_thresh):
                    matches.append(m.reshape(1, 2))
                else:
                    unmatched_detections.append(m[1])
                    unmatched_trackers.append(m[0])

            if (len(matches) == 0):
                matches = np.empty((0, 2), dtype=int)
            else:
                matches = np.concatenate(matches, axis=0)

            # print ("unmatched detection")
            # print (unmatched_detections)
            # print ("unmatched trackers")
            # print (unmatched_trackers)
            # print ("matched")
            # print (matches)

            # Track management and update
            # Update matched track and unmatched track
            for t, trk in enumerate((self.tracks_per_frame)):
                if (t not in unmatched_trackers):
                    d_detected = matches[np.where(matches[:, 0] == t)[0][0],1]
                    trk.update_matched(detections, d_detected, frame_num)
                else: #adding more info, adding prediction quality with respect to time stamp
                    trk.update_unmatched(self.trackstepoverlap, frame_num)

                # else: # i don't want any prediction result from Kalman filter


            # Initialize new track for new detection
            for t, trk in enumerate(unmatched_detections):
                obj = KalmanFilterObj(detections[trk], self.framecount, dim_x=7, dim_z=4)
                #  dim_x = for all the information contained by the object
                #  dim_z = for the bounding box coordinate prediction
                self.tracks_per_frame.append(obj)
                self.trackCount += 1
                # print ("created new tracks")

            # self.active_tracks = []

            i = len(self.tracks_per_frame)
            for trk in reversed(self.tracks_per_frame):
                # d = trk.retrieve_state()
                self.active_tracks.append(trk)

                i -= 1

                if (trk.time_since_update > self.max_trace_length):
                    self.tracks_per_frame.pop(i)
                    trk.remove_trk_final_predictions(self.max_trace_length)
                    self.alltracks.append(trk) # appending all the tracks that have left, trying to build the whole chunk of all tracks here
                    print ("removing one object here")
                    print (len(self.alltracks))

                # i -= 1



        return (self.active_tracks)

