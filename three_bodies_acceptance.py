import math
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import dispy
import shapely.geometry as sg

NBR_OF_CPUs = 5

#NODES = ["127.0.0.1"]
ip_addr = "192.168.0.8"
NODES = ["192.168.0.8", "192.168.0.22", "192.168.0.23", "192.168.0.18"]

proton_mass = 1.0
deuterium_mass = 2.0*proton_mass
speed_to_SI_cm = 978897.1372228841 #from sqrt(energy in ev / mass in amu)*100

nbr_events_per_ker = 1000

#NBR_EVENTS_KINETIC_ENERGY = 100
#NBR_EVENTS_VELOCITY = 100

kers =  np.linspace(0.01, 10, 100) #eV
#kers = np.array([5])
BEAM_ENERGY = 18000 #eV

#Velocity of center of mass in laboratory frame
V_cm = math.sqrt(2*BEAM_ENERGY/3)*speed_to_SI_cm*np.array([0,0,1])

#Generate one  possible combination of particles kinetic energies E_i such that
#E_cm = sum_i E_i
def gen_kinetic_energies(E_cm):

    kinetic_energies_cm = np.array([rand.uniform(0, E_cm),
    rand.uniform(0, E_cm), rand.uniform(0, E_cm)])

    #must scale e_i for sum_i e_i = E_cm
    scaling_factor = E_cm/np.sum(kinetic_energies_cm)
    kinetic_energies_cm = scaling_factor*kinetic_energies_cm

    return kinetic_energies_cm

#Generate one  possible velocity \vec{v} such that the associated kinetic
#energy is E = mass*v**2/2
#def gen_velocity(kinetic_energy, mass):
def gen_velocity():
    #v = math.sqrt(2*kinetic_energy/mass)
    # v = 1
    # #View velocity with given speed as a sphere
    # theta = rand.uniform(0, math.pi)
    # phi = rand.uniform(0, 2*math.pi)
    # sin_theta = math.sin(theta)
    # cos_phi = math.cos(phi)
    # sin_phi = math.sin(phi)
    # cos_theta = math.cos(theta)
    #
    # v_x = v*sin_theta*cos_phi
    # v_y = v*sin_theta*sin_phi
    # v_z = v*cos_theta
    # return np.array([v_x, v_y, v_z])
    size = 1
    v_x = rand.uniform(-size,size)
    v_y = rand.uniform(-size,size)
    v_z = rand.uniform(-size,size)
    return np.array([v_x, v_y, v_z])

def is_momentum_conservation_verified(v_1cm, v_2cm, v_3cm, speed_up_boundary):

    tolerance = 10**-2
    momentum_conserved = True

    for i in range(0,3):
        momentum_conserved = momentum_conserved and \
        abs(v_1cm[i]+v_2cm[i]+v_3cm[i]) <= speed_up_boundary*tolerance
    return momentum_conserved

def comp_velocities_lab_list(velocities_cm_list, V_cm):
    v_lab_l = []
    for v_in_cm in velocities_cm_list:
        v_lab_l.append(v_in_cm+V_cm)
    return v_lab_l

def gen_valid_events(params_tuple):

    rand.seed()

    (ker, mass, nbr_events, V_cm, speed_to_SI_cm) = params_tuple

    #speed_up_boundary = math.sqrt(2*ker/mass)

    kinetic_energies_list = []
    velocities_cm_list = []

    while(len(kinetic_energies_list) < nbr_events):

        #kinetic_energies = gen_kinetic_energies(ker)

        #v_1cm = gen_velocity(kinetic_energies[0], mass)
        #v_2cm = gen_velocity(kinetic_energies[1], mass)
        v_1cm = gen_velocity()
        v_2cm = gen_velocity()
        v_3cm = -v_1cm-v_2cm

        v_1cm_squared = v_1cm.dot(v_1cm)
        v_2cm_squared = v_2cm.dot(v_2cm)
        v_3cm_squared = v_3cm.dot(v_3cm)

        #if np.linalg.norm(v_1cm) <= 0.5 and np.linalg.norm(v_2cm) <= 0.5 \
        #and np.linalg.norm(v_3cm) <= 0.5:
        v_square = v_1cm_squared+v_2cm_squared+v_3cm_squared
        distribution_cut = 1

        if v_square < distribution_cut:

            kin_energy1 = mass*v_1cm_squared/2
            kin_energy2 = mass*v_2cm_squared/2
            kin_energy3 = mass*v_3cm_squared/2
            scale_kin_energies = ker/(kin_energy1+kin_energy2+kin_energy3)
            kin_energy1 = kin_energy1*scale_kin_energies
            kin_energy2 = kin_energy2*scale_kin_energies
            kin_energy3 = kin_energy3*scale_kin_energies
            kinetic_energies = np.array([kin_energy1, kin_energy2, kin_energy3])
            scale_velocities = math.sqrt(scale_kin_energies)
            v_1cm = v_1cm*scale_velocities
            v_2cm = v_2cm*scale_velocities
            v_3cm = v_3cm*scale_velocities
            #v_3cm = gen_velocity(kinetic_energies[2], mass)

            #if is_momentum_conservation_verified(v_1cm, v_2cm, v_3cm,
            #speed_up_boundary):
            kinetic_energies_list.append(kinetic_energies)
            velocities_cm_list.append((v_1cm*speed_to_SI_cm,
            v_2cm*speed_to_SI_cm, v_3cm*speed_to_SI_cm))

    return (kinetic_energies_list, velocities_cm_list,
    comp_velocities_lab_list(velocities_cm_list, V_cm))

def import_libs():
    import random as rand
    import numpy as np
    import math

    global rand, np

    return 0

def distributed_events_gen(nbr_events_per_ker):

    mass = deuterium_mass

    cluster = dispy.JobCluster(gen_valid_events, nodes=NODES,
    depends=[gen_kinetic_energies, gen_velocity,is_momentum_conservation_verified,
    comp_velocities_lab_list], setup=import_libs, ip_addr=ip_addr)

    jobs = []

    #Distribute workload among nodes:
    # - if one ker: distribute over possible kinetic energies generation
    # - if many kers: one ker per node
    local_kers = kers
    if len(local_kers) == 1:
        nbr_of_jobs = min(NBR_OF_CPUs*3, nbr_events_per_ker)
        local_kers = np.ones(nbr_of_jobs)*local_kers[0]

        for ker in local_kers:
            gen_valid_events_params = (ker, mass,
            nbr_events_per_ker/nbr_of_jobs, V_cm, speed_to_SI_cm)
            job = cluster.submit(gen_valid_events_params)
            jobs.append(job)
    else:
        for ker in local_kers:
            gen_valid_events_params = (ker, mass,
            nbr_events_per_ker, V_cm, speed_to_SI_cm)
            job = cluster.submit(gen_valid_events_params)
            jobs.append(job)

    #Retrieve results
    kin_energies_list = []
    velocities_cm_list = []
    velocities_lab_list = []
    ker_list = []

    job_idx = 0
    for job in jobs:
        (kin_energies_sublist, velocities_cm_sublist, velocities_lab_sublist) = job()
        kin_energies_list = itertools.chain(kin_energies_list,
        kin_energies_sublist)
        velocities_cm_list = itertools.chain(velocities_cm_list,
        velocities_cm_sublist)
        velocities_lab_list = itertools.chain(velocities_lab_list,
        velocities_lab_sublist)

        ker_sublist = list(np.ones(len(kin_energies_sublist))*local_kers[job_idx])
        ker_list = itertools.chain(ker_list, ker_sublist)
        job_idx = job_idx+1

    cluster.close()
    kin_energies_list = list(kin_energies_list)
    velocities_cm_list = list(velocities_cm_list)
    velocities_lab_list = list(velocities_lab_list)
    ker_list = list(ker_list)
    return (ker_list, kin_energies_list, velocities_cm_list, velocities_lab_list)

def dalitz_plot(kin_energies_list):

    dalitz_x = []
    dalitz_y = []

    for kin_energies in kin_energies_list:
        x = (kin_energies[1]-kin_energies[0])/math.sqrt(3.0)
        y = kin_energies[2]-1.0/3.0
        dalitz_x.append(x)
        dalitz_y.append(y)

    plt.scatter(dalitz_x,dalitz_y, s=0.5)
    plt.show()

#(kin_energies_list, velocities_cm_list) = distributed_events_gen()
#dalitz_plot(kin_energies_list)

#Length are in cm
det2_z = 326
origin = sg.Point(0,0,0)
det1_center = sg.Point(3,0,316)
det2_center = sg.Point(-3,0,det2_z)
det3_center = sg.Point(0,-1.25,292)
det1_radius = 2
det2_radius = 2
det3_radius = 1.25

def shapely_circle_perp_to_z(center, radius):
    center_np_array = np.array(center)
    z_coord = center_np_array[2]
    #Generate 2D circle in plane x-y
    center_x_y = sg.Point(center_np_array[0],center_np_array[1])
    circle_2D = center_x_y.buffer(radius)
    #From coords of circle in 2D add z component to obtaine circle in 3D
    circle_3D_coords = []
    for (x,y) in circle_2D.exterior.coords:
        circle_3D_coords.append((x,y,z_coord))
    circle_3D = sg.Polygon(circle_3D_coords)
    return circle_3D

det1_circle = shapely_circle_perp_to_z(det1_center, det1_radius)
det2_circle = shapely_circle_perp_to_z(det2_center, det2_radius)
det3_circle = shapely_circle_perp_to_z(det3_center, det3_radius)

#NB: a circle is a polygon in shapely
def list_of_pts_components_polygon(polygon):
    x_list = []
    y_list = []
    z_list = []
    for (x,y,z) in polygon.exterior.coords:
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return (x_list,y_list,z_list)

def list_of_pts_components_line(line):
    x_list = []
    y_list = []
    z_list = []
    for (x,y,z) in line.coords:
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return (x_list,y_list,z_list)

def plot_dets(det_circle_list, lines=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for det_circle in det_circle_list:
        (x_list, y_list, z_list) = list_of_pts_components_polygon(det_circle)
        ax.plot(x_list,y_list,z_list,c="blue")

    if lines is not None:
        for line in lines:
            (x_list, y_list, z_list) = list_of_pts_components_line(line)
            ax.plot(x_list,y_list,z_list)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

det_pt_furthest_from_origin = np.array(det2_center)+np.array(sg.Point(det2_radius,0,0))
max_dist_from_origin_to_det = np.linalg.norm(det_pt_furthest_from_origin)

def velocity_line_point_at_z(velocity,z):
    z_norm_vec = np.array([0,0,1])
    velocity_dir = velocity/np.linalg.norm(velocity)
    cos_theta = z_norm_vec.dot(velocity_dir)
    dist_to_xy_plane_perp_to_z_axis = z/cos_theta
    point_at_z = velocity_dir*dist_to_xy_plane_perp_to_z_axis
    return sg.Point(point_at_z)

# If particle never hits detector, return -1
# Otherwise returns the time needed for the particle to reach the detector
def detection_time(det_circle, velocity):

    #Velocity vector must be positive in its z component to (maybe) intersect
    #detector
    if velocity[2] <= 0:
        return -1

    #Check if line of director vector "velocity" intersect detector surface
    det_z_coord = det_circle.exterior.coords[0][2]
    point_at_z = velocity_line_point_at_z(velocity, det_z_coord)
    intersect = point_at_z.intersection(det_circle)

    if intersect.is_empty:
        return -1

    dist = np.linalg.norm(np.array(intersect))
    #print(dist)
    speed = np.linalg.norm(velocity)
    #print("speed "+str(speed))
    return dist/speed

def is_part_detected(det, velocity):
    return detection_time(det, velocity) != -1

# def is_part_detected(velocity, det_list):
#     det_idx = 0
#     for det in det_list:
#         if detection_time(det, velocity) != -1:
#             return det_idx
#         det_idx = det_idx+1
#     return -1


def is_detected(velocities_lab, det_list):

    v1 = velocities_lab[0]
    v2 = velocities_lab[1]
    v3 = velocities_lab[2]

    v1_det1 = is_part_detected(det_list[0], v1)
    v2_det1 = is_part_detected(det_list[0], v2)
    v3_det1 = is_part_detected(det_list[0], v3)
    v1_det2 = is_part_detected(det_list[1], v1)
    v2_det2 = is_part_detected(det_list[1], v2)
    v3_det2 = is_part_detected(det_list[1], v3)
    v1_det3 = is_part_detected(det_list[2], v1)
    v2_det3 = is_part_detected(det_list[2], v2)
    v3_det3 = is_part_detected(det_list[2], v3)

    return v1_det1 and v2_det2 and v3_det3 or \
    v1_det1 and v3_det2 and v2_det3 or \
    v1_det2 and v2_det1 and v3_det3 or \
    v1_det2 and v2_det3 and v3_det1 or \
    v1_det3 and v2_det1 and v3_det2 or \
    v1_det3 and v2_det2 and v3_det1


    # det_list_local = det_list
    # v1 = velocities_lab[0]
    # det1_idx = is_part_detected(v1, det_list_local)
    # if det1_idx == -1:
    #     return False
    # del det_list_local[det1_idx]
    # v2 = velocities_lab[1]
    # det2_idx = is_part_detected(v2, det_list_local)
    # if det2_idx == -1:
    #     return False
    # del det_list_local[det2_idx]
    # v3 = velocities_lab[2]
    # det3_idx = is_part_detected(v3, det_list_local)
    # return det3_idx != -1

def nbr_events_detected(params):
    velocities_lab_sublist, velocities_lab_idx, det_list = params
    detected_lab_idx = []
    count = 0
    local_idx = 0
    for velocities_lab in velocities_lab_sublist:
        if is_detected(velocities_lab, det_list):
            count = count+1
            detected_lab_idx.append(velocities_lab_idx[local_idx])
            #velocities_lab_idx = numpy.delete(velocities_lab_idx, local_idx)
        local_idx = local_idx+1
    return (count, detected_lab_idx)

def setup_node_nbr_events_detected():

    global sg, np

    import shapely.geometry as sg
    import numpy as np
    return 0



def distributed_nbr_events_detected(velocities_lab_list):

    det_list = [det1_circle,det2_circle,det3_circle]

    cluster = dispy.JobCluster(nbr_events_detected, nodes=NODES,
    depends=[is_detected,is_part_detected,detection_time,velocity_line_point_at_z],
    setup=setup_node_nbr_events_detected, ip_addr=ip_addr)
    #depends=[], setup=import_libs)
    nbr_of_jobs = NBR_OF_CPUs*3
    nbr_list_el_per_job = math.ceil(len(velocities_lab_list)/float(nbr_of_jobs))

    jobs = []
    for i in range(0, nbr_of_jobs):
        list_start_idx = i*nbr_list_el_per_job
        list_end_idx_plus_one = min((i+1)*nbr_list_el_per_job, len(velocities_lab_list))
        sub_list = velocities_lab_list[list_start_idx:list_end_idx_plus_one]
        velocities_lab_idx = np.arange(list_start_idx, list_end_idx_plus_one)
        if len(velocities_lab_idx) != 0:
            nbr_events_detected_params = (sub_list, velocities_lab_idx, det_list)
            job = cluster.submit(nbr_events_detected_params)
            jobs.append(job)

    total_count = 0
    velocities_lab_idx = []
    for job in jobs:
        (local_count, velocities_lab_idx_sublist) = job()
        velocities_lab_idx = itertools.chain(velocities_lab_idx,
        velocities_lab_idx_sublist)
        total_count = total_count+local_count
    cluster.close()
    return (total_count, velocities_lab_idx)

#gen_valid_events_params = (5, 1, 1, np.array([1,1,1]), 1)
#gen_valid_events(gen_valid_events_params)

(ker_list, kin_energies_list, velocities_cm_list, velocities_lab_list) = distributed_events_gen(nbr_events_per_ker)
#print(kin_energies_list)

#dalitz_plot(kin_energies_list)


def line_from_velocity(velocity, end_of_line_pt_z):
    point_at_z = velocity_line_point_at_z(velocity, end_of_line_pt_z)
    return sg.LineString([origin, point_at_z])

#Dispay all events
# for velocities in velocities_lab_list:
#     line1 = line_from_velocity(velocities[0], det2_z)
#     line2 = line_from_velocity(velocities[1], det2_z)
#     line3 = line_from_velocity(velocities[2], det2_z)
#     plot_dets([det1_circle,det2_circle,det3_circle], [line1,line2,line3])


nbr_total_event = len(velocities_lab_list)
(nbr_events_detected_var, velocities_lab_idx) = distributed_nbr_events_detected(velocities_lab_list)

#nbr_events_detected_params = (velocities_lab_list, np.arange(0, len(velocities_lab_list)), [det1_circle,det2_circle,det3_circle])
#(nbr_events_detected_var, velocities_lab_idx) = nbr_events_detected(nbr_events_detected_params)

print(nbr_total_event)
print(nbr_events_detected_var)

ker_events_nbr = {}

for idx in range(0,len(ker_list)):
    ker = ker_list[idx]
    if ker in ker_events_nbr:
        ker_events_nbr[ker] = ker_events_nbr[ker]+1
    else:
        ker_events_nbr[ker] = 1

ker_events_det = {}
for idx in velocities_lab_idx:
    ker = ker_list[idx]
    if ker in ker_events_det:
        ker_events_det[ker] = ker_events_det[ker]+1
    else:
        ker_events_det[ker] = 1

det_ratio = []

for key in ker_events_det.keys():
    det_ratio.append(ker_events_det[key]/ker_events_nbr[key])

plt.plot(ker_events_det.keys(),det_ratio)
plt.show()
# for idx in velocities_lab_idx:
#     line1 = line_from_velocity(velocities_lab_list[idx][0], det2_z)
#     line2 = line_from_velocity(velocities_lab_list[idx][1], det2_z)
#     line3 = line_from_velocity(velocities_lab_list[idx][2], det2_z)
#     plot_dets([det1_circle,det2_circle,det3_circle], [line1,line2,line3])


#print(detection_time(det1_circle, velocities_cm_list[0][0]))
#print(detection_time(det1_circle, np.array([0.05,0,1])))


#Code below plot energies distribution
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ker = 10
# for i in range(0,100):
#     E_kin = gen_kinetic_energies(ker)
#     ax.scatter(E_kin[0], E_kin[1], E_kin[2])
# plt.show()

#Code below plot velocities distribution
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# kinetic_energy = 10
# for i in range(0,100):
#     v = gen_velocity(kinetic_energy, deuterium_mass)
#     ax.scatter(v[0], v[1], v[2])
# plt.show()
