#!/usr/bin/env bash
# after yolo_photo.sh (pid 1610968): the dashboard's mission path flown once end to end -- the example mission to
# logan_pad_4 (markers 14, 13, 12), calm, the rule landing. One Gazebo at a time.
while kill -0 1610968 2>/dev/null; do sleep 30; done
cd /home/sai/RL_projects/rl-landing
echo "$(date +%H:%M) mission start" >> outputs/queue/yolo_photo.log
gnss_denied/fly_mission.sh outputs/gnss_denied/missions/example_pad4.json mission_example_pad4 > outputs/queue/mission_example_pad4.out 2>&1
echo "$(date +%H:%M) mission: $(tail -2 outputs/gazebo_geo/mission_example_pad4/run.log | tr '\n' ' ' | cut -c1-300)" >> outputs/queue/yolo_photo.log
