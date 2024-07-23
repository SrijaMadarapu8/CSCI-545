#!/usr/bin/env python

# Developed for USC 545 Intro To Robotics.

import rospy
import rosbag
import tf

import geometry_msgs.msg
import nav_msgs.msg
import sensor_msgs.msg
import std_msgs.msg

import argparse
import matplotlib.pyplot as plt
import numba
import numpy as np
import skimage.draw
import sys
import copy
import scipy.stats
LAMBDA_SHORT = 0.05

dtype = np.float64

LINEAR_MODEL_VAR_X = 0.5
LINEAR_MODEL_VAR_Y = 0.5
ANGULAR_MODEL_VAR = 0.3
SENSOR_MODEL_VAR = 15.0
NUM_PARTICLES = 200


@numba.jit(nopython=True)
def _CastRay(p0, p1, grid_data):
    '''
    Cast a ray from p0 in the direction toward p1 and return the first cell the
    sensor will detect.
    '''
    delta = np.abs(np.array([p1[0] - p0[0], p1[1] - p0[1]]))
    p = p0.copy()
    n = 1 + np.sum(delta)
    inc = np.array([1 if p1[0] > p0[0] else -1, 1 if p1[1] > p0[1] else -1])
    error = delta[0] - delta[1]
    delta *= 2

    last_p = p.copy()
    for i in range(n):
        if not (0 <= p[0] < grid_data.shape[1]
                and 0 <= p[1] < grid_data.shape[0]):
            return last_p
        last_p = p.copy()

        if grid_data[p[1], p[0]] != 0:  # coord flip for handedness
            return p

        if error > 0:
            p[0] += inc[0]
            error -= delta[1]
        else:
            p[1] += inc[1]
            error += delta[0]

    return p1


@numba.jit(nopython=True)
def _ComputeSimulatedRanges(scan_angles, scan_range_max, world_t_map,
                            map_t_particle, map_R_particle, grid_data,
                            grid_resolution):
    '''
    Cast rays in all directions and calculate distances to obstacles in all
    directions from a given particle.
    '''
    src_cell = ((map_t_particle - world_t_map) / grid_resolution).astype(
        np.int32)

    # Compute max possible cell indices
    angles = scan_angles + map_R_particle
    max_points = map_t_particle + scan_range_max * np.stack(
        (np.cos(angles), np.sin(angles)), axis=-1)
    max_cells = ((max_points - world_t_map) / grid_resolution).astype(np.int32)

    simulated_cell_ranges_sq = np.zeros_like(scan_angles)
    for i in range(len(max_cells)):
        hit_cell = _CastRay(src_cell, max_cells[i], grid_data)
        simulated_cell_ranges_sq[i] = np.sum(np.square(hit_cell - src_cell))

    return grid_resolution * np.sqrt(simulated_cell_ranges_sq)


@numba.jit(nopython=True)
def RotateBy(x, angle):
    c, s = np.cos(angle), np.sin(angle)
    m = np.zeros((2, 2), dtype=dtype)
    m[0, 0] = c
    m[0, 1] = -s
    m[1, 0] = s
    m[1, 1] = c
    return np.dot(m, x)


class Pose(object):
    '''
    2D pose representation of robot w.r.t. world frame.
    Given by the translation from origin to the robot, and then rotation to math
    robot orientation.
    '''
    def __init__(self, rotation=0, translation=[0, 0]):
        self.rotation = rotation
        self.translation = np.array(translation, dtype=dtype)

    @staticmethod
    def FromGeometryMsg(pose_msg):
        q = pose_msg.orientation
        tx = 1 - 2 * (q.y * q.y + q.z * q.z)
        ty = 2 * (q.x * q.y + q.z * q.w)
        yaw = np.arctan2(ty, tx)
        x, y = pose_msg.position.x, pose_msg.position.y
        return Pose(yaw, [x, y])

    def ToGeometryMsg(self):
        msg = geometry_msgs.msg.Pose()
        msg.position.x = self.translation[0]
        msg.position.y = self.translation[1]
        msg.orientation.z = np.sin(self.rotation / 2)
        msg.orientation.w = np.cos(self.rotation / 2)
        return msg


class Grid(object):
    '''
    Occupancy grid representation, with coordinate helpers.
    World is global coordinate frame and grid is index-based array coordinates.
    '''
    def __init__(self, grid_msg):
        self.cols = grid_msg.info.width
        self.rows = grid_msg.info.height
        self.resolution = grid_msg.info.resolution
        self.map_t_extents = [
            self.cols * self.resolution,
            self.rows * self.resolution,
        ]
        self.world_T_map = Pose.FromGeometryMsg(grid_msg.info.origin)
        self.data = np.asarray(grid_msg.data,
                               dtype=np.int8).reshape(self.rows, self.cols)

    def GridToWorld(self, cell_index):
        return self.resolution * cell_index + self.world_T_map.translation

    def WorldToGrid(self, point):
        return ((point - self.world_T_map.translation) /
                self.resolution).astype(np.int32)

    def GetWorldCoords(self, point):
        x, y = self.WorldToGrid(point)
        return self.data[y, x]

    def ToNavMsg(self):
        msg = nav_msgs.msg.OccupancyGrid()
        msg.info.resolution = self.resolution
        msg.info.width = self.cols
        msg.info.height = self.rows
        msg.info.origin = self.world_T_map.ToGeometryMsg()
        msg.data = self.data.flatten().tolist()
        return msg


class Scan(object):
    '''
    Scan representation / cache.
    Scan consists of angles and distances (ranges).
    '''
    def __init__(self, scan_msg):
        angle_min = scan_msg.angle_min
        angle_max = scan_msg.angle_max
        self.range_max = scan_msg.range_max

        self.angles = np.linspace(angle_min,
                                  angle_max,
                                  num=len(scan_msg.ranges))

        self.ranges = np.array(scan_msg.ranges, dtype=dtype)


class Particle(object):
    '''Individual particle representation.'''
    def __init__(self, grid, map_T_particle=None):
        if map_T_particle is None:
            ll = grid.world_T_map.translation
            ul = ll + grid.map_t_extents

            found = False
            while not found:
                translation = np.random.uniform(ll, ul)
                if grid.GetWorldCoords(translation) == 0:
                    break

            map_T_particle = Pose(rotation=np.random.uniform(-np.pi, np.pi),
                                  translation=translation)

        self.grid = grid
        self.map_T_particle = map_T_particle
        self.last_odom_timestamp = None

    def UpdateOdom(self, odom_msg):
        '''Propagate this particle according to the sensor data and motion
        model.'''
        if self.last_odom_timestamp is not None:
            dt = (odom_msg.header.stamp - self.last_odom_timestamp).to_sec()
        else:
            dt = 0
        self.last_odom_timestamp = odom_msg.header.stamp

        ##########
        #
        #  YOUR CODE HERE (Odometry Section)
        #
        #  1. Extract the particle's velocity in its local frame
        #     (odom_msg.twist.twist.linear.{x, y})
        vx = odom_msg.twist.twist.linear.x
        vy = odom_msg.twist.twist.linear.y

        #  2. Add noise to the velocity, with component variance
        #     LINEAR_MODEL_VAR_X and LINEAR_MODEL_VAR_Y
        vx += np.random.normal(0, LINEAR_MODEL_VAR_X**0.5)
        vy += np.random.normal(0, LINEAR_MODEL_VAR_Y**0.5)

        #  3. Transform the linear velocity into map frame (use the provided
        #     RotateBy() function). The current pose of the particle, in map
        #     frame, is stored in self.map_T_particle
        local_velocity = np.array([vx, vy], dtype=dtype)
        world_velocity = RotateBy(local_velocity, self.map_T_particle.rotation)

        #  4. Integrate the linear velocity to the particle pose, stored in
        #     self.map_T_particle.translation
        self.map_T_particle.translation += world_velocity * dt

        #  5. Extract the particle's rotational velocity
        #     (odom_msg.twist.twist.angular.z)
        omega = odom_msg.twist.twist.angular.z

        #  6. Add noise to the rotational velocity, with variance
        #     ANGULAR_MODEL_VAR
        omega += np.random.normal(0, ANGULAR_MODEL_VAR**0.5)

        #  7. Integrate the rotational velocity into the particle pose, stored
        #     in self.map_T_particle.rotation
        self.map_T_particle.rotation += omega * dt
        #
        ##########

    def _ComputeSimulatedRanges(self, scan):
        translation = self.map_T_particle.translation
        rotation = self.map_T_particle.rotation
        return _ComputeSimulatedRanges(scan.angles, scan.range_max,
                                       self.grid.world_T_map.translation,
                                       translation, rotation, self.grid.data,
                                       self.grid.resolution)

    def learn_intrinsic_parameters(self, scan, iter=1000):
    
        para = [0, 0, 0, 0, SENSOR_MODEL_VAR, LAMBDA_SHORT]
        sim_ranges = self._ComputeSimulatedRanges(scan)
        for j in range(iter):
            sum_e_hit, sum_e_short, sum_e_max, sum_e_rand = 0, 0, 0, 0
            spe_sum1, spe_sum2 = 0, 0

            for i in range(len(scan.angles)):
                true_range = scan.ranges[i]
                sim_range = sim_ranges[i]

                local_error = self.local_measurement_noise_error(true_range, sim_range, scan.range_max, para[4])
                unexpected_error = self.unexpected_objects_error(true_range, sim_range, para[5])
                failure_error = self.failures_error(true_range, scan.range_max)
                random_error = self.random_measurements_error(true_range, scan.range_max)

                total_err = local_error + unexpected_error + failure_error + random_error

                if total_err == 0:
                    e_hit = 0
                    e_short = 0
                    e_max = 0
                    e_rand = 0
                else:
                    e_hit = local_error / total_err
                    e_short = unexpected_error / total_err
                    e_max = failure_error / total_err
                    e_rand = random_error / total_err

                sum_e_hit += e_hit
                sum_e_short += e_short
                sum_e_max += e_max
                sum_e_rand += e_rand

                spe_sum1 += e_hit * (true_range - sim_range)**2
                spe_sum2 += e_short * true_range
                print(spe_sum2)
            if abs((sum_e_hit / len(scan.angles)) - para[0]) <= 0.01 and abs((sum_e_short / len(scan.angles)) - para[1]) <= 0.01 and abs((sum_e_max / len(scan.angles)) - para[2]) <= 0.01 and abs((sum_e_rand / len(scan.angles)) - para[3]) <= 0.01 and abs(((spe_sum1 / sum_e_hit)**0.5) - para[4]) <= 0.01 and abs((sum_e_short / spe_sum2) - para[5]) <= 0.01:
                break
            para[0] = sum_e_hit / len(scan.angles)
            para[1] = sum_e_short / len(scan.angles)
            para[2] = sum_e_max / len(scan.angles)
            para[3] = sum_e_rand / len(scan.angles)
            para[4] = (spe_sum1 / sum_e_hit)**0.5
            para[5] = sum_e_short / spe_sum2
        return para

    def local_measurement_noise_error(self, true_range, sim_range, range_max, std_hit):
        if true_range < 0 or true_range > range_max:
            return 0

        err = np.exp(-0.5 * (true_range - sim_range)**2 / std_hit**2) / (std_hit * np.sqrt(2 * np.pi))
        err /= scipy.integrate.quad(
                lambda x: np.exp(-0.5 * (true_range - x)**2 / std_hit**2) / (std_hit * np.sqrt(2 * np.pi)),
                0,
                range_max
            )[0]
        return err

    def unexpected_objects_error(self, true_range, sim_range, lambda_short=0.05):
        if true_range < 0 or true_range > sim_range:
            return 0

        err = lambda_short * np.exp(-1 * lambda_short * sim_range)
        err /= 1 - np.exp(-1 * lambda_short * true_range)
        return err

    def failures_error(self, true_range, range_max):
        if true_range == range_max:
            return 1
        else:
            return 0

    def random_measurements_error(self, true_range, range_max):
        if true_range < 0 or true_range >= range_max:
            return 0
        return 1 / range_max

    def UpdateScan(self, scan):
        '''
        Calculate weights of particles according to scan / map matching.
        '''
        
        sim_ranges = self._ComputeSimulatedRanges(scan)
        # z_hit, z_short, z_max, z_rand, var_hit, lambda_short = self.learn_intrinsic_parameters(scan)
        z_hit, z_short, z_max, z_rand, var_hit, lambda_short = 0.43, 0.2, 0.01, 0.36, 4.7, 0.6
        # print(z_hit, z_short, z_max, z_rand, var_hit, lambda_short)
        # exit(1)
        ##########
        #
        #  YOUR CODE HERE (LIDAR Section)
        #
        #  1. Compute the likelihood of each beam, compared to sim_ranges. The
        #     true measurement vector is in scan.ranges. Use the Gaussian PDF
        #     formulation.
        self.weight = 1
        for i in range(len(scan.angles)):
            true_range = scan.ranges[i]
            sim_range = sim_ranges[i]
            

            # Calculate the Gaussian probability density function
            # You can use numpy's Gaussian PDF function for this
            local_error = self.local_measurement_noise_error(true_range, sim_range, scan.range_max, var_hit)
            unexpected_error = self.unexpected_objects_error(true_range, sim_range, lambda_short)
            failure_error = self.failures_error(true_range, scan.range_max)
            random_error = self.random_measurements_error(true_range, scan.range_max)
            
            probability = z_hit * local_error + z_short * unexpected_error + z_max * failure_error + z_rand * random_error
            if probability == 0:
                continue
            self.weight *= probability

        #  2. Assign the weight of this particle as the product of these
        #     probabilities. Store this value in self.weight.
        #
        ##########


class ParticleFilter(object):
    def __init__(self, grid, num_particles):
        self.grid = grid
        self.last_timestamp = None
        self.particles = [Particle(grid) for _ in range(num_particles)]
        self.pose_publisher = rospy.Publisher("pose_hypotheses",
                                              geometry_msgs.msg.PoseArray,
                                              queue_size=10)
        self.scan_publisher = rospy.Publisher("lidar",
                                              sensor_msgs.msg.LaserScan,
                                              queue_size=10)
        self.tf_broadcaster = tf.TransformBroadcaster()

    def GetMeanPose(self):
        sum_translation = np.zeros(2, dtype=dtype)
        sum_rotation = 0
        for particle in self.particles:
            sum_translation += particle.map_T_particle.translation
            sum_rotation += particle.map_T_particle.rotation

        avg_translation = sum_translation / len(self.particles)
        avg_rotation = sum_rotation / len(self.particles)

        return Pose(avg_rotation, avg_translation)

    def GetPoseArray(self):
        msg = geometry_msgs.msg.PoseArray()
        msg.poses = [
            particle.map_T_particle.ToGeometryMsg()
            for particle in self.particles
        ]
        msg.header.stamp = self.last_timestamp
        msg.header.frame_id = "map"

        return msg

    def UpdateOdom(self, odom_msg):
        if self.last_timestamp is None:
            self.last_timestamp = odom_msg.header.stamp

        for particle in self.particles:
            particle.UpdateOdom(odom_msg)

    def UpdateScan(self, scan_msg):
        if self.last_timestamp is None:
            self.last_timestamp = scan_msg.header.stamp

        scan = Scan(scan_msg)

        # Update weights for particles.
        for particle in self.particles:
            particle.UpdateScan(scan)

        ##########
        #
        #  YOUR CODE HERE (Importance Sampling Section)
        #
        #  1. Each particle has a weight, i.e. self.particles[i].weight
        weights = [particle.weight for particle in self.particles]

        #  2. Use these weights to build a normalized discrete probability
        #     distribution for sampling each particle
        normalized_weights = np.array(weights, dtype=dtype) / sum(weights)

        #  3. Use np.random.choice to choose the indices of the particles that
        #     we want to propagate to the next step (use the p= argument to pass
        #     your probability distribution)
        num_particles = len(self.particles)
        selected_indices = np.random.choice(num_particles, size=num_particles, p=normalized_weights)

        #  4. Use copy.deepcopy to copy the proper particles into a new population
        #     of particles.
        new_particles = [copy.deepcopy(self.particles[index]) for index in selected_indices]
       
        #  5. Set self.particles to the new population of particles.
        self.particles = new_particles
        ##########

        # Publish cloud for visualization.
        self.pose_publisher.publish(self.GetPoseArray())
        avg_pose = self.GetMeanPose()
        avg_trans = (avg_pose.translation[0], avg_pose.translation[1], 0)
        avg_quat = tf.transformations.quaternion_from_euler(
            0, 0, avg_pose.rotation)
        self.tf_broadcaster.sendTransform(avg_trans, avg_quat,
                                          scan_msg.header.stamp, "robot",
                                          "map")
        self.scan_publisher.publish(scan_msg)


def main():
    # rospy initialization.
    rospy.init_node('usc545mcl')
    args = rospy.myargv(argv=sys.argv)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_particles', type=int, default=NUM_PARTICLES)
    argparser.add_argument('bag_filename', type=str)
    args = argparser.parse_args(args[1:])

    # Wait for map server to start up.
    print("Waiting for map_server...")
    grid_msg = rospy.wait_for_message("/map", nav_msgs.msg.OccupancyGrid)
    print("Map received.")

    # Construct particle filter for the received map.
    grid = Grid(grid_msg)
    mcl = ParticleFilter(grid, args.num_particles)

    # "Subscribe" to message channels from bag.
    # You can see the relevant channels by running
    # $ rosbag info /path/to/bag
    bag = rosbag.Bag(args.bag_filename)
    subscribers = {"lidar": mcl.UpdateScan, "odom": mcl.UpdateOdom}
    x_err = []
    y_err = []
    yaw_err = []
    for topic, msg, t in bag.read_messages(topics=["gt_odom"] +
                                           subscribers.keys()):
        if topic == "gt_odom":
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            yaw = 2 * np.arcsin(msg.pose.pose.orientation.z)

            mcl_pose = mcl.GetMeanPose()
            x_err.append((x - mcl_pose.translation[0])**2)
            y_err.append((y - mcl_pose.translation[1])**2)
            yaw_err.append((yaw - mcl_pose.rotation)**2)

        else:
            subscribers[topic](msg)

    plt.plot(x_err, label="x_mse")
    plt.plot(y_err, label="y_mse")
    plt.plot(yaw_err, label="yaw_mse")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()