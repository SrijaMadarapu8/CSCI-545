#!/usr/bin/env python
"""
BSD 3-Clause License
	
	Copyright (c) 2018, Personal Robotics Laboratory
	All rights reserved.
	
	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are met:
	
	* Redistributions of source code must retain the above copyright notice, this
	  list of conditions and the following disclaimer.
	
	* Redistributions in binary form must reproduce the above copyright notice,
	  this list of conditions and the following disclaimer in the documentation
	  and/or other materials provided with the distribution.
	
	* Neither the name of the copyright holder nor the names of its
	  contributors may be used to endorse or promote products derived from
	  this software without specific prior written permission.
	
	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
	FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
	DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
	SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
	CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
	OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import rospy
from std_msgs.msg import String

def publish_topic():
    # Create a ROS publisher that publishes to the topic 'sensor_reading'. It should give a String message, with a queue size of 10
    ### START CODE HERE ###
    pub = rospy.Publisher('sensor_reading', String, queue_size=10)
    ### END CODE HERE ###

    # Create a node 'ros_node_publisher'
    ### START CODE HERE ###
    rospy.init_node('ros_node_publisher', anonymous=True)
    ### END CODE HERE ###
        
    # Set a publishing rate of 10 hz (10 messages per second)
    ### START CODE HERE ###
    rate = rospy.Rate(10)
    ### END CODE HERE ###

    while not rospy.is_shutdown():
        # Write string formatted as 'Y <time>' where time is the ros time since start
        # We always return a Y as the message, meaning the obstacle is always detected
	### START CODE HERE ###
        msg = 'Y %s' % rospy.get_time()
	### END CODE HERE ###

        # Log and publish the information to the 'sensor_reading' topic
	### START CODE HERE ###
        rospy.loginfo(msg)
        pub.publish(msg)
        rate.sleep()
	### END CODE HERE ###    

if __name__ == '__main__':
    try:
        publish_topic()
    except rospy.ROSInterruptException as e:
        print('ROS exception {0}'.format(e))
        pass
