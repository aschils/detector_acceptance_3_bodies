import math
import numpy as np
import random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import dispy
import shapely.geometry as sg
from scipy.stats import gaussian_kde

NBR_OF_CPUs = 200

NODES = ["192.168.0.*"]
#ip_addr = "192.168.0.15"
port=10001
scheduler_node = "192.168.0.15"
#NODES = ["192.168.0.8", "192.168.0.22", "192.168.0.23", "192.168.0.18",
#"192.168.0.11"]

proton_mass = 1.0
deuterium_mass = 2.0*proton_mass
speed_to_SI_cm = 978897.1372228841 #from sqrt(energy in ev / mass in amu)*100

nbr_events_per_ker = 100 #15000

kers =  np.linspace(3, 3, 500) #eV
#kers = np.linspace(6, 10, 333)
BEAM_ENERGY = 18000 #eV

#Velocity of center of mass in laboratory frame
V_cm = math.sqrt(BEAM_ENERGY/3)*speed_to_SI_cm*np.array([0,0,1])

def gen_velocity():
    size = 1
    v_x = rand.uniform(-size,size)
    v_y = rand.uniform(-size,size)
    v_z = rand.uniform(-size,size)
    return np.array([v_x, v_y, v_z])

def comp_velocities_lab_list(velocities_cm_list, V_cm):
    v_lab_l = []
    for v_in_cm in velocities_cm_list:
        v_lab_l.append(v_in_cm+V_cm)
    return v_lab_l

def gen_valid_events(params_tuple):

    rand.seed()
    (ker, mass, nbr_events, V_cm, speed_to_SI_cm) = params_tuple
    kinetic_energies_list = []
    velocities_cm_list = []

    while(len(kinetic_energies_list) < nbr_events):
        v_1cm = gen_velocity()
        v_2cm = gen_velocity()
        v_3cm = -v_1cm-v_2cm
        v_1cm_squared = v_1cm.dot(v_1cm)
        v_2cm_squared = v_2cm.dot(v_2cm)
        v_3cm_squared = v_3cm.dot(v_3cm)
        v_square = v_1cm_squared+v_2cm_squared+v_3cm_squared
        distribution_cut = 1

        #Allows to obtain a uniform distribution for v_3
        #Distribution not initially unifrom because sum of two random uniform
        #variables is not an uniform distribution.
        if v_square < distribution_cut:
            kin_energy1 = v_1cm_squared
            kin_energy2 = v_2cm_squared
            kin_energy3 = v_3cm_squared
            kin_energies_sum = kin_energy1+kin_energy2+kin_energy3

            #if kin_energies_sum > 0:
            scale_kin_energies = ker/kin_energies_sum
            kin_energy1 = kin_energy1*scale_kin_energies
            kin_energy2 = kin_energy2*scale_kin_energies
            kin_energy3 = kin_energy3*scale_kin_energies
            kinetic_energies = np.array([kin_energy1, kin_energy2, kin_energy3])
            scale_velocities = math.sqrt(scale_kin_energies)
            v_1cm = v_1cm*scale_velocities
            v_2cm = v_2cm*scale_velocities
            v_3cm = v_3cm*scale_velocities

            kinetic_energies_list.append(kinetic_energies)
            velocities_cm_list.append((v_1cm*speed_to_SI_cm,
            v_2cm*speed_to_SI_cm, v_3cm*speed_to_SI_cm))
    return (kinetic_energies_list, velocities_cm_list,
    comp_velocities_lab_list(velocities_cm_list, V_cm))

def setup_gen_valid_events():
    import random as rand
    import numpy as np
    import math

    global rand, np

    return 0

def distributed_events_gen(nbr_events_per_ker, local_kers):

    mass = deuterium_mass

    cluster = dispy.SharedJobCluster(gen_valid_events, nodes=NODES,
    depends=[gen_velocity, comp_velocities_lab_list],
    setup=setup_gen_valid_events, port=port)#, ip_addr=ip_addr)

    jobs = []

    #Distribute workload among nodes:
    # - if one ker: distribute over possible kinetic energies generation
    # - if many kers: one ker per node

    if len(local_kers) == 1:
        nbr_of_jobs = min(NBR_OF_CPUs*3, nbr_events_per_ker)
        local_kers = np.ones(nbr_of_jobs)*local_kers[0]

        effective_nbr_events_per_ker = nbr_events_per_ker/nbr_of_jobs* \
        len(local_kers)

        for ker in local_kers:
            gen_valid_events_params = (ker, mass,
            nbr_events_per_ker/nbr_of_jobs, V_cm, speed_to_SI_cm)
            job = cluster.submit(gen_valid_events_params)
            jobs.append(job)
    else:
        effective_nbr_events_per_ker = nbr_events_per_ker
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
        job_res = job()
        if job_res is not None:
            (kin_energies_sublist, velocities_cm_sublist, velocities_lab_sublist) = job_res
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
    return (ker_list, kin_energies_list, velocities_cm_list, velocities_lab_list,
    effective_nbr_events_per_ker)

def dalitz_plot(kin_energies_list, show=True):

    dalitz_x = []
    dalitz_y = []

    for kin_energies in kin_energies_list:
        x = (kin_energies[1]-kin_energies[0])/math.sqrt(3.0)
        y = kin_energies[2]-1.0/3.0
        dalitz_x.append(x)
        dalitz_y.append(y)

    plt.scatter(dalitz_x,dalitz_y, s=0.5)
    if show:
        plt.show()


#(kin_energies_list, velocities_cm_list) = distributed_events_gen()
#dalitz_plot(kin_energies_list)

#Length are in cm
det1_z = 316
det2_z = 326
det3_z = 292
det3_x = 0.4
det3_y = 1.75
origin = sg.Point(0,0,0)
det1_center = sg.Point(2.9,0,det1_z)
det2_center = sg.Point(-2.98,0,det2_z)
det3_center = sg.Point(det3_x,det3_y,det3_z)
det1_radius = 1.92
det2_radius = 1.9
det3_radius = 1.25
time_ref_det_id = 1 #det1 is used for delta time measurements

def z_pos_of_det(det_id):
    if det_id == 0:
        return det1_z
    elif det_id == 1:
        return det2_z
    elif det_id == 2:
        return det3_z
    else:
        return None

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
det_list = [det1_circle,det2_circle,det3_circle]

#NB: a circle is a polygon in shapely
def list_of_pts_components_polygon(polygon):
    x_list = []
    y_list = []
    z_list = []

    ext_pts = polygon.exterior.coords
    if len(ext_pts[0]) == 3:
        for (x,y,z) in polygon.exterior.coords:
            x_list.append(x)
            y_list.append(y)
            z_list.append(z)
        return (x_list,y_list,z_list)
    else:
        for (x,y) in polygon.exterior.coords:
            x_list.append(x)
            y_list.append(y)
        return (x_list,y_list)

def list_of_pts_components_line(line):
    x_list = []
    y_list = []
    z_list = []
    for (x,y,z) in line.coords:
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return (x_list,y_list,z_list)

def plot_dets(det_circle_list, lines=None, lines_colors=None, circles2D=None,
circles2D_z=None):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for det_circle in det_circle_list:
        (x_list, y_list, z_list) = list_of_pts_components_polygon(det_circle)
        ax.plot(x_list,y_list,z_list,c="blue")

    if lines is not None:
        if lines_colors is not None:
            plt.gca().set_color_cycle(lines_colors)
        line_idx = 0
        for line in lines:
            (x_list, y_list, z_list) = list_of_pts_components_line(line)
            ax.plot(x_list,y_list,z_list)
        line_idx = line_idx + 1

    if circles2D is not None:
        for i in range(0, len(circles2D)):
            (x_list, y_list) = list_of_pts_components_polygon(circles2D[i])
            ax.plot(x_list,y_list,circles2D_z[i],c="green")

    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
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

def detection_info(det_circle_list, det_idx, velocity):

    #Velocity vector must be positive in its z component to (maybe) intersect
    #detector
    if velocity[2] <= 0:
        return None
    det_circle = det_circle_list[det_idx]
    #Check if line of director vector "velocity" intersect detector surface
    det_z_coord = det_circle.exterior.coords[0][2]
    point_at_z = velocity_line_point_at_z(velocity, det_z_coord)
    intersect = point_at_z.intersection(det_circle)

    if intersect.is_empty:
        return None

    dist = np.linalg.norm(np.array(intersect))
    speed = np.linalg.norm(velocity)
    time = dist/speed
    intersect_np = np.array(intersect)
    X = intersect_np[0]
    Y = intersect_np[1]
    det_id = det_idx+1
    return [X, Y, time, det_id]

def is_part_detected(det_info):
    return det_info is not None

def is_detected(velocities_lab, det_list):

    v1 = velocities_lab[0]
    v2 = velocities_lab[1]
    v3 = velocities_lab[2]

    v1_det1_info = detection_info(det_list, 0, v1)
    v2_det1_info = detection_info(det_list, 0, v2)
    v3_det1_info = detection_info(det_list, 0, v3)
    v1_det2_info = detection_info(det_list, 1, v1)
    v2_det2_info = detection_info(det_list, 1, v2)
    v3_det2_info = detection_info(det_list, 1, v3)
    v1_det3_info = detection_info(det_list, 2, v1)
    v2_det3_info = detection_info(det_list, 2, v2)
    v3_det3_info = detection_info(det_list, 2, v3)

    v1_det1 = is_part_detected(v1_det1_info)
    v2_det1 = is_part_detected(v2_det1_info)
    v3_det1 = is_part_detected(v3_det1_info)
    v1_det2 = is_part_detected(v1_det2_info)
    v2_det2 = is_part_detected(v2_det2_info)
    v3_det2 = is_part_detected(v3_det2_info)
    v1_det3 = is_part_detected(v1_det3_info)
    v2_det3 = is_part_detected(v2_det3_info)
    v3_det3 = is_part_detected(v3_det3_info)

    if v1_det1 and v2_det2 and v3_det3:
        return (v1_det1_info, v2_det2_info, v3_det3_info)
    if v1_det1 and v3_det2 and v2_det3:
        return (v1_det1_info, v3_det2_info, v2_det3_info)
    if v1_det2 and v2_det1 and v3_det3:
        return (v1_det2_info, v2_det1_info, v3_det3_info)
    if v1_det2 and v2_det3 and v3_det1:
        return (v1_det2_info, v2_det3_info, v3_det1_info)
    if v1_det3 and v2_det1 and v3_det2:
        return (v1_det3_info, v2_det1_info, v3_det2_info)
    if v1_det3 and v2_det2 and v3_det1:
        return (v1_det3_info, v2_det2_info, v3_det1_info)
    return None

def nbr_events_detected(params):
    velocities_lab_sublist, velocities_lab_idx, det_list = params
    detected_lab_idx = []
    det_info_list = []
    count = 0
    local_idx = 0
    for velocities_lab in velocities_lab_sublist:
        det_info = is_detected(velocities_lab, det_list)
        if det_info is not None:
            count = count+1
            detected_lab_idx.append(velocities_lab_idx[local_idx])
            #velocities_lab_idx = numpy.delete(velocities_lab_idx, local_idx)
            det_info_list.append(det_info)
        local_idx = local_idx+1
    return (count, detected_lab_idx, det_info_list)

def setup_node_nbr_events_detected():

    global sg, np

    import shapely.geometry as sg
    import numpy as np
    return 0


def distributed_nbr_events_detected(velocities_lab_list):

    det_list = [det1_circle,det2_circle,det3_circle]

    cluster = dispy.SharedJobCluster(nbr_events_detected, nodes=NODES,
    depends=[is_detected,is_part_detected,detection_info,velocity_line_point_at_z],
    setup=setup_node_nbr_events_detected, port=port)#, ip_addr=ip_addr)

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
    velocities_lab_idx = {}
    det_info_list = []
    for job in jobs:
        (local_count, velocities_lab_idx_sublist, det_info_sublist) = job()

        for detected_velocities_lab_idx in velocities_lab_idx_sublist:
            velocities_lab_idx[detected_velocities_lab_idx] = True

        det_info_list = itertools.chain(det_info_list,
        det_info_sublist)
        total_count = total_count+local_count
    cluster.close()
    return (total_count, velocities_lab_idx, list(det_info_list))

def line_from_velocity(velocity, end_of_line_pt_z):
    point_at_z = velocity_line_point_at_z(velocity, end_of_line_pt_z)
    return sg.LineString([origin, point_at_z])

#Debug purpose
# for ker in kers:
#     gen_valid_events_params = (ker, deuterium_mass, nbr_events_per_ker, V_cm, speed_to_SI_cm)
#     (kin_energies_list, velocities_cm_list, velocities_lab_list) = \
#     gen_valid_events(gen_valid_events_params)

#(ker_list, kin_energies_list, velocities_cm_list, velocities_lab_list,
#effective_nbr_events_per_ker) = distributed_events_gen(nbr_events_per_ker, kers)
#print(kin_energies_list)

#dalitz_plot(kin_energies_list)



#nbr_total_event = len(velocities_lab_list)
#print("nbr_total_event "+str(nbr_total_event))

#nbr_events_detected_params = (velocities_lab_list, np.arange(0, len(velocities_lab_list)), [det1_circle,det2_circle,det3_circle])
#(nbr_events_detected_var, velocities_lab_idx, det_info_list) = nbr_events_detected(nbr_events_detected_params)

#(nbr_events_detected_var, velocities_lab_idx, det_info_list) = \
#distributed_nbr_events_detected(velocities_lab_list)
#print("nbr_events_detected_var "+str(nbr_events_detected_var))

# det_kin_energies = []
#
# for idx in velocities_lab_idx:
#     det_kin_energies.append(kin_energies_list[idx])
# dalitz_plot(det_kin_energies)



#Compute number of events generated
# ker_events_nbr = {}
# for idx in range(0,len(ker_list)):
#     ker = ker_list[idx]
#     if ker in ker_events_nbr:
#         ker_events_nbr[ker] = ker_events_nbr[ker]+1
#     else:
#         ker_events_nbr[ker] = 1

# ker_events_det = {}
# for idx in velocities_lab_idx:
#     ker = ker_list[idx]
#     if ker in ker_events_det:
#         ker_events_det[ker] = ker_events_det[ker]+1
#     else:
#         ker_events_det[ker] = 1
#
# det_ratio = []
# for ker in ker_list:
#    det_ratio.append(float(ker_events_det.get(ker, 0))/effective_nbr_events_per_ker)

# det_ratio_file = open("det_ratio.txt", "w")
# det_ratio_file.write(str(ker_events_det.keys())+"\n")
# det_ratio_file.write(str(det_ratio))
# det_ratio_file.close()
# plt.plot(ker_list,det_ratio)
# plt.show()
#
# counter = 1
# for idx in velocities_lab_idx:
#     if nbr_events_detected_var % counter == math.ceil(nbr_events_detected_var/5):
#         line1 = line_from_velocity(velocities_lab_list[idx][0], det2_z)
#         line2 = line_from_velocity(velocities_lab_list[idx][1], det2_z)
#         line3 = line_from_velocity(velocities_lab_list[idx][2], det2_z)
#         plot_dets([det1_circle,det2_circle,det3_circle], [line1,line2,line3])
#     counter = counter+1

def to_exp_measurements(det_info_list):

    exp_measurements = []

    for det_infos in det_info_list:

        det_info1, det_info2, det_info3 = det_infos
        Xp1, Yp1, tp1, det_id1 = det_info1
        Xp2, Yp2, tp2, det_id2 = det_info2
        Xp3, Yp3, tp3, det_id3 = det_info3

        #t1: detection time at detector 1
        #t2: detection time at detector 2
        #t3: detection time at detector 3

        if det_id1 == 1:
            t1 = tp1
            X1 = Xp1
            Y1 = Yp1
            if det_id2 == 2:
                t2 = tp2
                X2 = Xp2
                Y2 = Yp2
                t3 = tp3
                X3 = Xp3
                Y3 = Yp3
            else:
                t2 = tp3
                X2 = Xp3
                Y2 = Yp3
                t3 = tp2
                X3 = Xp2
                Y3 = Yp2
        elif det_id1 == 2:
            t2 = tp1
            X2 = Xp1
            Y2 = Yp1
            if det_id2 == 1:
                t1 = tp2
                X1 = Xp2
                Y1 = Yp2
                t3 = tp3
                X3 = Xp3
                Y3 = Yp3
            else:
                t1 = tp3
                X1 = Xp3
                Y1 = Yp3
                t3 = tp2
                X3 = Xp2
                Y3 = Yp2
        else:
            t3 = tp1
            X3 = Xp1
            Y3 = Yp1
            if det_id2 == 1:
                t1 = tp2
                X1 = Xp2
                Y1 = Yp2
                t2 = tp3
                X2 = Xp3
                Y2 = Yp3
            else:
                t2 = tp2
                X2 = Xp2
                Y2 = Yp2
                t1 = tp3
                X1 = Xp3
                Y1 = Yp3

        dT12 = t2-t1
        dT13 = t3-t1

        exp_measurements.append((X1,Y1,X2,Y2,X3,Y3,dT12,dT13,t1))

    return exp_measurements

def write_sim_results(det_info_list):
    out_f = open("sim_results.txt", "w")

    out_f.write("center of mass velocity along X,Y,Z:\n")
    out_f.write(str(V_cm[0])+" "+str(V_cm[1])+" "+str(V_cm[2])+"\n")
    out_f.write("X1 Y1 X2 Y2 X3 Y3 dT12 dT13 t1\n")

    exp_measurements = to_exp_measurements(det_info_list)

    for exp_measurement in exp_measurements:
        (X1,Y1,X2,Y2,X3,Y3,dT12,dT13,t1) = exp_measurement
        out_f.write(str(X1)+" "+str(Y1)+" "+str(X2)+" "+str(Y2)+" "+str(X3)+\
        " "+str(Y3)+" "+str(dT12)+" "+str(dT13)+" "+str(t1)+"\n")

    out_f.close()

def compute_acceptance(ker_list, velocities_lab_idx, effective_nbr_events_per_ker):

    ker_events_det = {}
    for idx in velocities_lab_idx:
        ker = ker_list[idx]
        if ker in ker_events_det:
            ker_events_det[ker] = ker_events_det[ker]+1
        else:
            ker_events_det[ker] = 1

    det_ratio = []
    for ker in ker_list:
       det_ratio.append(float(ker_events_det.get(ker, 0))/effective_nbr_events_per_ker)
    return det_ratio

def run_simulation():

    acceptance = []
    #kers_for_hist = []
    det_info_list = []

    for ker in kers:
        #(ker_list, kin_energies_list, velocities_cm_list, velocities_lab_list,
        #effective_nbr_events_per_ker) = distributed_events_gen(
        #nbr_events_per_ker, [ker])

        gen_valid_events_params = (ker, deuterium_mass, nbr_events_per_ker, V_cm, speed_to_SI_cm)
        (kin_energies_list, velocities_cm_list, velocities_lab_list) = \
             gen_valid_events(gen_valid_events_params)


        #Dispay all events
        # for velocities in velocities_lab_list:
        #      line1 = line_from_velocity(velocities[0], det2_z)
        #      line2 = line_from_velocity(velocities[1], det2_z)
        #      line3 = line_from_velocity(velocities[2], det2_z)
        #      plot_dets([det1_circle,det2_circle,det3_circle], [line1,line2,line3])


        #(nbr_events_detected_var, velocities_lab_idx, det_info_list) = \
        #distributed_nbr_events_detected(velocities_lab_list)
        nbr_events_detected_params = (velocities_lab_list,
        np.arange(0, len(velocities_lab_list)), [det1_circle,det2_circle,det3_circle])
        (nbr_events_detected_var, velocities_lab_idx, det_info_sublist) = nbr_events_detected(nbr_events_detected_params)

        for i in velocities_lab_idx:
            velocities = velocities_lab_list[i]
            line1 = line_from_velocity(velocities[0], det2_z)
            line2 = line_from_velocity(velocities[1], det2_z)
            line3 = line_from_velocity(velocities[2], det2_z)
            plot_dets([det1_circle,det2_circle,det3_circle], [line1,line2,line3])


        #kers_for_hist = itertools.chain(kers_for_hist, nbr_events_detected_var*[ker])
        det_info_list = itertools.chain(det_info_list, det_info_sublist)

        acceptance_l = nbr_events_detected_var/nbr_events_per_ker
        acceptance.append(acceptance_l)
        #compute_acceptance(ker, velocities_lab_idx,
        #nbr_events_per_ker)
        #acceptance = itertools.chain(acceptance, acceptance_sub_list)

    det_info_list = list(det_info_list)
    write_sim_results(det_info_list)

    #kers_for_hist = list(kers_for_hist)
    #print(kers_for_hist)
    #plt.hist(kers_for_hist, 10000)
    #plt.show()

    acceptance = list(acceptance)
    det_ratio_file = open("det_ratio.txt", "w")
    det_ratio_file.write(str(kers)+"\n")
    det_ratio_file.write(str(acceptance))
    det_ratio_file.close()
    plt.plot(kers,acceptance)
    plt.show()


#run_simulation()

# def velocity_in_circle(velocity, radius, det_id):
#     det_z = z_pos_of_det(det_id)
#     pt_at_z = np.array(velocity_line_point_at_z(velocity, det_z))
#     x = pt_at_z[0]
#     y = pt_at_z[1]
#     pt_at_z = np.array([x,y])
#     #print("pt_at_z "+str(pt_at_z))
#     #print("np.linalg.norm(pt_at_z) "+str(np.linalg.norm(pt_at_z)))
#     #print("radius "+str(radius))
#
#     return np.linalg.norm(pt_at_z) <= radius
#
#     #pt_radius = np.linalg.norm(np.array(pt_at_z))
#
# def impact_circle(ker, number_of_events):
#
#     gen_valid_events_params = (ker, deuterium_mass, number_of_events, V_cm,
#     speed_to_SI_cm)
#     (kin_energies_list, velocities_cm_list, velocities_lab_list) = \
#     gen_valid_events(gen_valid_events_params)
#
#     nbr_events_detected_params = (velocities_lab_list,
#     np.arange(0, len(velocities_lab_list)), [det1_circle,det2_circle,det3_circle])
#     (nbr_events_detected_var, velocities_lab_idx, det_info_list) = \
#     nbr_events_detected(nbr_events_detected_params)
#
#     center = sg.Point(0,0)
#     init_radius = 0.1
#     radii = [init_radius, init_radius, init_radius]
#     circles = [center.buffer(init_radius), center.buffer(init_radius),
#     center.buffer(init_radius)]
#     radius_scale_factor = 1.01
#
#     c = 0
#     print("velocities_lab_idx "+str(len(velocities_lab_idx)))
#     for idx in velocities_lab_idx:
#         velocities = velocities_lab_list[idx]
#         det_info_sl = det_info_list[c]
#         c = c+1
#         for i in range(0,len(velocities)):
#             det_info = det_info_sl[i]
#             det_idx = det_info[3]-1
#
#             while not velocity_in_circle(velocities[i], radii[det_idx], det_idx):
#                 radii[det_idx] = radii[det_idx]*radius_scale_factor
#                 circles[det_idx] = center.buffer(radii[det_idx])
#
#     return (circles, radii)
#
#
#
#
#
#
# def max_cone_angle_and_max_Y12(ker, number_of_events):
#
# #(ker_list, kin_energies_list, velocities_cm_list, velocities_lab_list,
# #effective_nbr_events_per_ker) = distributed_events_gen(
# #nbr_events_per_ker, [ker])
#
#     gen_valid_events_params = (ker, deuterium_mass, number_of_events, V_cm,
#     speed_to_SI_cm)
#     (kin_energies_list, velocities_cm_list, velocities_lab_list) = \
#     gen_valid_events(gen_valid_events_params)
#
#     nbr_events_detected_params = (velocities_lab_list,
#     np.arange(0, len(velocities_lab_list)), [det1_circle,det2_circle,det3_circle])
#     (nbr_events_detected_var, velocities_lab_idx, det_info_list) = \
#     nbr_events_detected(nbr_events_detected_params)
#
#
#
#     #(ker_list, kin_energies_list, velocities_cm_list, velocities_lab_list,
#     #effective_nbr_events_per_ker) = distributed_events_gen(number_of_events,
#     #[ker])
#     #(nbr_events_detected_var, velocities_lab_idx, det_info_list) = \
#     #distributed_nbr_events_detected(velocities_lab_list)
#
#     max_angle = 0
#
#     for idx in velocities_lab_idx:
#         velocities = velocities_lab_list[idx]
#         for velocity in velocities:
#             angle = math.acos(velocity[2]/np.linalg.norm(velocity))
#             if angle > max_angle:
#                 max_angle = angle
#
#     exp_measurements = to_exp_measurements(det_info_list)
#
#     max_Y = 0
#
#     for exp_meas in exp_measurements:
#         (X1,Y1,X2,Y2,X3,Y3,dT12,dT13,t1) = exp_meas
#         if Y1 > max_Y:
#             max_Y = Y1
#         if Y2 > max_Y:
#             max_Y =Y2
#
#     return (max_angle, max_Y)
