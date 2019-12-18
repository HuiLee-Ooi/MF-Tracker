video='rouen stmarc sherbrooke rene'
for vid in $video
do
    python3 Run_tracker_det_only.py UrbanTracker RetinaNet $vid
done
