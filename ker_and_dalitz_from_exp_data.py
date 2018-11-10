import numpy as np
import math
import three_bodies_acceptance as tba
import shapely.geometry as sg
import matplotlib.pyplot as plt
import itertools
import random as rand
import scipy as sp
from scipy.fftpack import fft, ifft
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp

L1 = 316 #cm
L2 = 326 #cm
L3 = 292 #cm detector for which we don't have position (X3,Y3)

speed_to_SI_cm = 978897.1372228841 #from sqrt(energy in ev / mass in amu)*100

def noise_for_hist_bar(noise_dT12, noise_X1, noise_Y1, noise_X2, noise_Y2,
noise_dT13, noise_hist, noise_hist_bar_idx):

    subl_left_idx = 0
    for i in range(0,noise_hist_bar_idx):
        subl_left_idx = subl_left_idx+noise_hist[i]
    subl_right_idx = subl_left_idx+noise_hist[noise_hist_bar_idx]-1

    return (noise_dT12[subl_left_idx:subl_right_idx+1],
    noise_X1[subl_left_idx:subl_right_idx+1],
    noise_Y1[subl_left_idx:subl_right_idx+1],
    noise_X2[subl_left_idx:subl_right_idx+1],
    noise_Y2[subl_left_idx:subl_right_idx+1],
    noise_dT13[subl_left_idx:subl_right_idx+1])

# n_dT12 = []
# n_X1 = []
# n_Y1 = []
# n_X2 = []
# n_Y2 = []
# n_dT13 = []
#
# j1 = 0
# j2 = 0
# for i in range(0, len(noise_dT12_hist)):
#     if noise_dT12_hist[i] == 0:
#         noise_dT12_hist[i] = noise_dT12_hist_not_0[j1]
#         (dT12l, X1l, Y1l, X2l, Y2l, dT13l) = noise_for_hist_bar(noise_dT12_hist_not_0,
#         j1)
#         dT12l = (bins[i]+(bins[i+1]-bins[i])/2.0)*np.ones(noise_dT12_hist[i])
#         j1 = (j1+1) % len(noise_dT12_hist_not_0)
#     else:
#         (dT12l, X1l, Y1l, X2l, Y2l, dT13l) = noise_for_hist_bar(noise_dT12_hist_not_0,
#         j2)
#         dT12l = (bins[i]+(bins[i+1]-bins[i])/2.0)*np.ones(noise_dT12_hist[i])
#         j2 = j2+1
#     n_dT12 = itertools.chain(n_dT12, dT12l)
#     n_X1 = itertools.chain(n_X1, X1l)
#     n_Y1 = itertools.chain(n_Y1, Y1l)
#     n_X2 = itertools.chain(n_X2, X2l)
#     n_Y2 = itertools.chain(n_Y2, Y2l)
#     n_dT13 = itertools.chain(n_dT13, dT13l)
#
# n_dT12 = np.array(list(n_dT12))
# n_X1 = np.array(list(n_X1))
# n_Y1 = np.array(list(n_Y1))
# n_X2 = np.array(list(n_X2))
# n_Y2 = np.array(list(n_Y2))
# n_dT13 = np.array(list(n_dT13))

# plt.hist(n_dT12, 300)
# plt.title("n_dT12")
# plt.show()
#
# plt.hist(n_X1, 300)
# plt.title("n_X1")
# plt.show()
#
# plt.hist(n_Y1, 300)
# plt.title("n_Y1")
# plt.show()
#
# plt.hist(n_X2, 300)
# plt.title("n_X2")
# plt.show()
#
# plt.hist(n_Y2, 300)
# plt.title("n_Y2")
# plt.show()
#
# plt.hist(n_dT13, 300)
# plt.title("n_dT13")
# plt.show()


#print(noise_dT12_hist)

#dT12_shifted1 = dT12 +1.6*10**-7
#dT12_shifted2 = dT12 -3.5*10**-7

def gaussian(x,a,x0,sigma):
    return a*exp(-(x-x0)**2/(2*sigma**2))

def centers_of_bins(bins):
    centers = np.zeros(len(bins)-1)
    for i in range(0, centers.size):
        centers[i] = bins[i] + (bins[i+1]-bins[i])/2.0
    return centers

def filter_wrt_dT(exp_data, dT, max_diff_tol, fit_dT_bounds=None):

    nbr_bars = 300

    bins_bounds = np.linspace(np.min(dT), np.max(dT), nbr_bars)
    dT_hist, bins = np.histogram(dT, bins_bounds)
    bin_iv = bins_bounds[1]-bins_bounds[0]
    bins_centers = centers_of_bins(bins)

    if fit_dT_bounds is None:
        binsf = bins
        dT_histf = dT_hist
        bins_boundsf = bins_bounds
    else:
        lb = fit_dT_bounds[0]
        rb = fit_dT_bounds[1]
        dTf = dT[dT > lb]
        dTf = dTf[dTf < rb]

        #bins_boundsf = np.linspace(np.min(dTf), np.max(dTf), nbr_bars)
        bins_boundsf = bins_bounds
        dT_histf, binsf = np.histogram(dTf, bins_boundsf)

    bin_ivf = bins_boundsf[1]-bins_boundsf[0]
    bins_centersf = centers_of_bins(binsf)
    a = max(dT_histf)
    mean = sum(bins_centersf*dT_histf)/sum(dT_histf)
    sigma = math.sqrt(sum(dT_histf*(bins_centersf-mean)**2)/sum(dT_histf))
    popt,pcov = curve_fit(gaussian, bins_centersf, dT_histf, p0=[a,mean,sigma])

    dT_hist_fitted = gaussian(bins_centers,*popt)

    plt.plot(bins_centers, dT_hist, 'b+:',label='data')
    plt.plot(bins_centers, dT_hist_fitted, 'ro:',label='fit')
    plt.show()

    hist_diff = abs(dT_hist_fitted-dT_hist)/dT_hist_fitted
    #max_diff_tol = 0.2
    exp_data_filt = []
    col_nbr = exp_data.shape[1]
    for j in range(0,col_nbr):
        bin_idx = math.floor((dT[j]-bins[0])/bin_iv)
        if bin_idx == len(bins)-1:
            bin_idx = bin_idx-1

        if hist_diff[bin_idx] <= max_diff_tol:
            exp_data_filt.append(exp_data[:,j])
    return np.array(exp_data_filt)

def remove_noisy_events(X1, X2, Y1, Y2, dT12, dT13):

    exp_data = np.array([X1, Y1, X2, Y2, dT12, dT13])
    exp_data = filter_wrt_dT(exp_data, dT12, 0.2, (0, 3*10**-7))
    dT13 = exp_data[:,5]
    exp_data = np.transpose(exp_data)
    exp_data = filter_wrt_dT(exp_data, dT13, 0.2, (-4.7*10**-7, -1.4*10**-7))
    return exp_data

# plt.hist(dT12, 300)
# plt.title("dT12 filtered")
# plt.show()
#
# plt.hist(dT13, 300)
# plt.title("dT13 filtered")
# plt.show()


def t1_roots(dT12, dT13, V_cm_norm):
    alpha = dT12*dT13
    beta = dT12+dT13
    a = np.complex_(-3*V_cm_norm)
    b = np.complex_(L1+L2+L3-3*V_cm_norm*beta)
    c = np.complex_(L1*beta+dT13*L2+dT12*L3-3*V_cm_norm*alpha)
    d = np.complex_(alpha*L1)

    #Compute roots of polynom at**3 + bt**2 + ct + d = 0

    t1_1 = (np.sqrt((-27*a**2*d + 9*a*b*c - 2*b**3)**2 + 4*(3*a*c - b**2)**3) \
    - 27*a**2*d + 9*a*b*c - 2*b**3)**(1.0/3)/(3*2**(1.0/3)*a) \
    - (2**(1.0/3)*(3*a*c - b**2))/(3*a*(np.sqrt((-27*a**2*d + 9*a*b*c - 2*b**3)**2 \
    + 4*(3*a*c - b**2)**3) - 27*a**2*d + 9*a*b*c - 2*b**3)**(1.0/3)) - b/(3*a)

    t1_2 = -((1 - 1j*np.sqrt(3))*(np.sqrt((-27*a**2*d + 9*a*b*c - 2*b**3)**2 + \
    4*(3*a*c - b**2)**3) - 27*a**2*d + 9*a*b*c - 2*b**3)**(1.0/3))/(6*2**(1.0/3)*a) + \
    ((1 + 1j*np.sqrt(3))*(3*a*c - b**2))/(3*2**(2.0/3)*a* \
    (np.sqrt((-27*a**2*d + 9*a*b*c - 2*b**3)**2 + 4*(3*a*c - b**2)**3) - 27*a**2*d + \
    9*a*b*c - 2*b**3)**(1.0/3)) - b/(3.0*a)

    t1_3 = -((1 + 1j*np.sqrt(3))*(np.sqrt((-27*a**2*d + 9*a*b*c - 2*b**3)**2 + \
    4*(3*a*c - b**2)**3) - 27*a**2*d + 9*a*b*c - 2*b**3)**(1.0/3))/(6*2**(1.0/3)*a) +\
    ((1 - 1j*np.sqrt(3))*(3*a*c - b**2))/(3.0*2**(2.0/3)*a*\
    (np.sqrt((-27*a**2*d + 9*a*b*c - 2*b**3)**2 + 4*(3*a*c - b**2)**3)\
    - 27*a**2*d + 9*a*b*c - 2*b**3)**(1.0/3)) - b/(3.0*a)

    return (t1_1, t1_2, t1_3)

def compute_X3_and_Y3(X1, Y1, X2, Y2, t1, dT12, dT13):
    X3 = -(t1+dT13)*(X1/t1+X2/(t1+dT12))
    Y3 = -(t1+dT13)*(Y1/t1+Y2/(t1+dT12))

    return (X3, Y3)

def is_X3_Y3_on_det3(X3, Y3):
    intersect = sg.Point(X3, Y3).intersection(tba.det3_circle)
    return not intersect.is_empty
    #return (X3-.4)**2+(Y3-1.75)**2 < 1.25**2

def max_angle_with_Z(ker):
    part_kin_energy_cm = ker/2
    v_cm = math.sqrt(2*part_kin_energy_cm/tba.deuterium_mass)*speed_to_SI_cm*np.array([1,0,0])
    V = v_cm+V_cm
    return math.acos(V[2]/np.linalg.norm(V))


#max_angle = max_angle_with_Z(5)
#(max_angle, max_Y) = tba.max_cone_angle_and_max_Y12(13, 2000)

# simu_circles = tba.impact_circle(13, 1000)
# print(simu_circles)
# z_pos_of_dets = [tba.z_pos_of_det(0), tba.z_pos_of_det(1), tba.z_pos_of_det(2)]
# tba.plot_dets(det_list, simu_circles, z_pos_of_dets)

#print("max_angle degré "+str(max_angle/math.pi*180))

#treshold_l = np.linspace(0.0001, 0.001, 1000)
#nbr_events_kept = []

def keep_valid_events(X1, Y1, X2, Y2, dT12, dT13, V_cm):

    V_cm_norm = np.linalg.norm(V_cm)
    (t1_1, t1_2, t1_3) = t1_roots(dT12, dT13, V_cm_norm)

    #for th in treshold_l:
    th = 0.0015
    #th = 0.0005
    #th = 1
    t1_list = []
    events_idx_to_keep = np.zeros(len(X1),  dtype=bool)

    for i in range(0,len(X1)):

        t1 = np.array([t1_1[i], t1_2[i], t1_3[i]])

        X3, Y3 = compute_X3_and_Y3(X1[i], Y1[i], X2[i], Y2[i], t1, dT12[i], dT13[i])
        X3 = np.real(X3)
        Y3 = np.real(Y3)
        t2 = t1+dT12[i]
        t3 = t1+dT13[i]

        sum_kin_energies = (X1[i]/t1)**2 + (Y1[i]/t1)**2 + (L1/t1)**2 + \
        (X2[i]/t2)**2 + (Y2[i]/t2)**2 + (L2/t2)**2 + (X3/t3)**2 + (Y3/t3)**2 + \
        (L3/t3)**2
        center_of_mass_kin_energy = 3*V_cm[2]**2
        energy_conservation = abs(sum_kin_energies-center_of_mass_kin_energy)
        best_root_idx = np.argmin(energy_conservation)
        energy_conservation = energy_conservation[best_root_idx]
        is_energy_conserved = energy_conservation/sum_kin_energies[best_root_idx] <= th
        #best_root_idx = np.argmin(abs(t1-316/V_cm_norm))


        t1_best = np.real(t1[best_root_idx])

        #V1_local = np.array([X1[i], Y1[i], L1])/t1_best
        #V2_local = np.array([X2[i], Y2[i], L2])/(t1_best+dT12[i])
        #V3_local = np.array([X3[best_root_idx], Y3[best_root_idx], L3])/(t1_best+dT13[i])

        #V1_angle = math.acos(V1_local[2]/np.linalg.norm(V1_local))
        #print("V1_angle "+str(V1_angle/math.pi*180))
        #V2_angle = math.acos(V2_local[2]/np.linalg.norm(V2_local))
        #V3_angle = math.acos(V3_local[2]/np.linalg.norm(V3_local))

        #print("V2_angle "+str(V2_angle/math.pi*180))

        valid_event = is_X3_Y3_on_det3(X3[best_root_idx], Y3[best_root_idx])# and \
        #is_energy_conserved #and tba.velocity_in_circle(V1_local, simu_circles[0], 0) \
        #and tba.velocity_in_circle(V2_local, simu_circles[1], 1) and \
        #tba.velocity_in_circle(V3_local, simu_circles[2], 2)

        #and V1_angle <= max_angle and V2_angle <= max_angle \
        #and V3_angle <= max_angle # and Y1[i] <= max_Y and Y2[i] <= max_Y

        if valid_event:
            t1_list.append(np.real(t1_best))
            events_idx_to_keep[i] = True

            #print("t1_best "+str(t1_best))
            #print("t1_vcm "+str(316/V_cm_norm))

        #nbr_events_kept.append(len(events_idx_to_keep[events_idx_to_keep]))

    #plt.plot(treshold_l, nbr_events_kept)
    #plt.show()

    #print((t1_1[events_idx_to_keep][-1],t1_2[events_idx_to_keep][-1],t1_3[events_idx_to_keep][-1]))

    t1 = np.array(t1_list)

    print("events_idx_to_keep ratio: "+str(len(events_idx_to_keep[events_idx_to_keep])/len(events_idx_to_keep)))

    X1 = X1[events_idx_to_keep]
    Y1 = Y1[events_idx_to_keep]
    X2 = X2[events_idx_to_keep]
    Y2 = Y2[events_idx_to_keep]
    dT12 = dT12[events_idx_to_keep]
    dT13 = dT13[events_idx_to_keep]
    X3, Y3 = compute_X3_and_Y3(X1, Y1, X2, Y2, t1, dT12, dT13)

    return (X1,Y1,X2,Y2,X3,Y3,dT12,dT13,t1)

def compute_v_lab(t1, X1, Y1, X2, Y2, dT12, X3, Y3, dT13):

    V1 = []
    V2 = []
    V3 = []
    for i in range(0, len(t1)):
        V1.append(np.array([X1[i], Y1[i], L1])/t1[i])
        V2.append(np.array([X2[i], Y2[i], L2])/(t1[i]+dT12[i]))
        V3.append(np.array([X3[i], Y3[i], L3])/(t1[i]+dT13[i]))
    V1 = np.array(V1)
    V2 = np.array(V2)
    V3 = np.array(V3)

    return (V1, V2, V3)

def events_lines_to_det(det_idx, V):
    det_z_coord = tba.det_list[det_idx].exterior.coords[0][2]
    lines = []
    for i in range(0,len(V)):
        line = tba.line_from_velocity(V[i], det_z_coord)
        lines.append(line)
    return lines

def plot_events(V1, V2, V3):
    color_det_1 = "b"
    color_det_2 = "r"
    color_det_3 = "g"
    lines = events_lines_to_det(0, V1)
    lines = itertools.chain(lines, events_lines_to_det(1, V2))
    lines = list(itertools.chain(lines, events_lines_to_det(2, V3)))
    lines_1_colors = [color_det_1]*len(V1)
    lines_2_colors = [color_det_2]*len(V2)
    lines_3_colors = [color_det_3]*len(V3)
    lines_colors = itertools.chain(lines_1_colors, lines_2_colors)
    lines_colors = list(itertools.chain(lines_colors, lines_3_colors))
    tba.plot_dets(tba.det_list, lines, lines_colors)

def plot_impacts(V1, V2, V3):


    x = []
    y = []

    for v in V1:
        pt = tba.velocity_line_point_at_z(v, tba.det_list[0].exterior.coords[0][2])
        pt = np.array(pt)
        x.append(pt[0])
        y.append(pt[1])

    for v in V2:
        pt = tba.velocity_line_point_at_z(v, tba.det_list[1].exterior.coords[0][2])
        pt = np.array(pt)
        x.append(pt[0])
        y.append(pt[1])

    for v in V3:
        pt = tba.velocity_line_point_at_z(v, tba.det_list[2].exterior.coords[0][2])
        pt = np.array(pt)
        x.append(pt[0])
        y.append(pt[1])

    plt.plot(x,y, 'r+')

    plt.show()


#plot_events(V1, V2, V3)

def compute_ker(V1, V2, V3, V_cm):

    v1_cm = (V1-V_cm)/speed_to_SI_cm
    v2_cm = (V2-V_cm)/speed_to_SI_cm
    v3_cm = (V3-V_cm)/speed_to_SI_cm

    total_ker = np.zeros(len(V1))
    kerp1 = np.zeros(len(V1))
    kerp2 = np.zeros(len(V1))
    kerp3 = np.zeros(len(V1))

    for i in range(0, len(v1_cm)):
        v1_cm_squared = v1_cm[i].dot(v1_cm[i])
        v2_cm_squared = v2_cm[i].dot(v2_cm[i])
        v3_cm_squared = v3_cm[i].dot(v3_cm[i])
        kerp1[i] = v1_cm_squared
        kerp2[i] = v2_cm_squared
        kerp3[i] = v3_cm_squared
        total_ker[i] = v1_cm_squared+v2_cm_squared+v3_cm_squared #eV

    return (total_ker, kerp1, kerp2, kerp3)

def ker_of_gen_noise(X1, X2, Y1, Y2, dT12, dT13, V_cm_norm):

    noise_dT12_bool1 = dT12 < -4*10**-8
    noise_dT12_bool2 = dT12 > -1.5*10**-7
    noise_dT12_bool = np.logical_and(noise_dT12_bool1, noise_dT12_bool2)
    noise_dT12 = dT12[noise_dT12_bool]
    noise_X1 = X1[noise_dT12_bool]
    noise_Y1 = Y1[noise_dT12_bool]
    noise_X2 = X2[noise_dT12_bool]
    noise_Y2 = Y2[noise_dT12_bool]
    noise_dT13 = dT13[noise_dT12_bool]

    sorted_by_dT12_idx = np.argsort(noise_dT12)
    noise_dT12 = noise_dT12[sorted_by_dT12_idx]
    noise_X1 = X1[sorted_by_dT12_idx]
    noise_Y1 = Y1[sorted_by_dT12_idx]
    noise_X2 = X2[sorted_by_dT12_idx]
    noise_Y2 = Y2[sorted_by_dT12_idx]
    noise_dT13 = dT13[sorted_by_dT12_idx]

    # plt.hist(noise_dT12, 300)
    # plt.title("noise_dT12")
    # plt.show()
    noise_dT12_hist, bins = np.histogram(noise_dT12, np.linspace(np.min(dT12), np.max(dT12), nbr_bars))
    noise_dT12_hist_not_0 = noise_dT12_hist[noise_dT12_hist > 0]

    n_dT12 = []
    n_X1 = []
    n_Y1 = []
    n_X2 = []
    n_Y2 = []
    n_dT13 = []

    j1 = 0
    j2 = 0
    for i in range(0, len(noise_dT12_hist)):
        if noise_dT12_hist[i] == 0:
            noise_dT12_hist[i] = noise_dT12_hist_not_0[j1]
            (dT12l, X1l, Y1l, X2l, Y2l, dT13l) = noise_for_hist_bar(noise_dT12,
            noise_X1, noise_Y1, noise_X2, noise_Y2,
            noise_dT13,noise_dT12_hist_not_0,
            j1)
            dT12l = (bins[i]+(bins[i+1]-bins[i])/2.0)*np.ones(noise_dT12_hist[i])
            j1 = (j1+1) % len(noise_dT12_hist_not_0)
        else:
            (dT12l, X1l, Y1l, X2l, Y2l, dT13l) = noise_for_hist_bar(noise_dT12,
            noise_X1, noise_Y1, noise_X2, noise_Y2,
            noise_dT13,noise_dT12_hist_not_0,
            j2)
            dT12l = (bins[i]+(bins[i+1]-bins[i])/2.0)*np.ones(noise_dT12_hist[i])
            j2 = j2+1
        n_dT12 = itertools.chain(n_dT12, dT12l)
        n_X1 = itertools.chain(n_X1, X1l)
        n_Y1 = itertools.chain(n_Y1, Y1l)
        n_X2 = itertools.chain(n_X2, X2l)
        n_Y2 = itertools.chain(n_Y2, Y2l)
        n_dT13 = itertools.chain(n_dT13, dT13l)

    n_dT12 = np.array(list(n_dT12))
    n_X1 = np.array(list(n_X1))
    n_Y1 = np.array(list(n_Y1))
    n_X2 = np.array(list(n_X2))
    n_Y2 = np.array(list(n_Y2))
    n_dT13 = np.array(list(n_dT13))

    # plt.hist(n_dT12, 300)
    # plt.title("n_dT12")
    # plt.show()
    #
    # plt.hist(n_X1, 300)
    # plt.title("n_X1")
    # plt.show()
    #
    # plt.hist(n_Y1, 300)
    # plt.title("n_Y1")
    # plt.show()
    #
    # plt.hist(n_X2, 300)
    # plt.title("n_X2")
    # plt.show()
    #
    # plt.hist(n_Y2, 300)
    # plt.title("n_Y2")
    # plt.show()
    #
    # plt.hist(n_dT13, 300)
    # plt.title("n_dT13")
    # plt.show()

    (X1vs1,Y1vs1,X2vs1,Y2vs1,X3vs1,Y3vs1,dT12vs1,dT13vs1,t1vs1) = \
    keep_valid_events(n_X1, n_Y1, n_X2, n_Y2, n_dT12, n_dT13, V_cm)
    (V1, V2, V3) = compute_v_lab(t1vs1, X1vs1, Y1vs1, X2vs1, Y2vs1, dT12vs1, X3vs1, Y3vs1, dT13vs1)
    ker_list_noise = compute_ker(V1, V2, V3, V_cm)
    return ker_list_noise

def acceptance_correction(kers, nbr_events, acceptance_kers, acceptance,
acceptance_f):

    corrected_nbr_events = np.zeros(len(kers))

    for j in range(0, len(kers)):
        ker = kers[j]
        i = np.abs(ker-acceptance_kers).argmin()

        a = 0
        b = 0
        delta_ker = 0
        if ker <= acceptance_kers[i] and i != 0:
            delta_ker = ker-acceptance_kers[i-1]
            delta_x = acceptance_kers[i]-acceptance_kers[i-1]
            if delta_x != 0:
                a = (acceptance[i]-acceptance[i-1])/delta_x
                b = acceptance[i-1]
        elif i != len(acceptance_kers)-1:
            delta_x = acceptance_kers[i+1]-acceptance_kers[i]
            a = (acceptance[i+1]-acceptance[i])/delta_x
            b = acceptance[i+1]
            delta_ker = ker-acceptance_kers[i]

        accept = a*delta_ker+b
        if accept > 0:
            corrected_nbr_events[j] = nbr_events[j]/accept
        else:
            corrected_nbr_events[j] = 0

    #return corrected_nbr_events
    return nbr_events/acceptance_f(kers)

def rotate(dT12, dT13, rot_mat):
    dT12_r = dT12*rot_mat[0,0]+dT13*rot_mat[0,1]
    dT13_r = dT12*rot_mat[1,0]+dT13*rot_mat[1,1]
    return (dT12_r, dT13_r)

def noise_dT12_dT13(X1, Y1, X2, Y2, dT12, dT13):

    theta = -30 #degré
    theta = theta/360.0*2*math.pi
    rot_mat = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta),
    math.cos(theta)]])

    dT12_r, dT13_r = rotate(dT12, dT13, rot_mat)
    point_size = abs(dT12[10]*dT13[10])
    sg_x_left = -2.2*10**-7
    sg_x_right = 1.32*10**-7
    sg_y_bot = -4.2*10**-7
    sg_y_up = -2.4*10**-7
    sg_x_to_keep = np.logical_and(dT12_r > sg_x_left, dT12_r < sg_x_right)
    sg_y_to_keep = np.logical_and(dT13_r > sg_y_bot, dT13_r < sg_y_up)
    sig_bloc_to_keep = np.logical_and(sg_x_to_keep, sg_y_to_keep)
    #sig_bloc_dT12_r = dT12_r[sig_bloc_to_keep]
    #sig_bloc_dT13_r = dT13_r[sig_bloc_to_keep]
    sig_bloc_dT12 = dT12[sig_bloc_to_keep]
    sig_bloc_dT13 = dT13[sig_bloc_to_keep]
    sig_bloc_X1 = X1[sig_bloc_to_keep]
    sig_bloc_Y1 = Y1[sig_bloc_to_keep]
    sig_bloc_X2 = X2[sig_bloc_to_keep]
    sig_bloc_Y2 = Y2[sig_bloc_to_keep]

    signal_plus_noise = (sig_bloc_X1, sig_bloc_Y1, sig_bloc_X2, sig_bloc_Y2,
    sig_bloc_dT12, sig_bloc_dT13)

    #plt.scatter(dT12, dT13, s=point_size, color="b")
    #plt.scatter(sig_bloc_dT12, sig_bloc_dT13, s=point_size, color="g")
    #plt.scatter(noise_bu_dT12, noise_bu_dT13, s=point_size, color="r")
    #plt.show()

    noises_bu = []
    noise_bu_x_left = -9*10**-8
    noise_bu_x_right = 3.33*10**-7
    noise_bu_y_bot = -1.05*10**-6
    delta_y = 3.6*10**-7
    delta_y_up = delta_y/10
    while noise_bu_y_bot+delta_y < sg_y_bot:
        noise_bu_y_up = noise_bu_y_bot+delta_y

        noise_bu_x_to_keep = np.logical_and(dT12 > noise_bu_x_left, dT12 < noise_bu_x_right)
        noise_bu_y_to_keep  = np.logical_and(dT13 > noise_bu_y_bot, dT13 < noise_bu_y_up)
        noise_bu_to_keep = np.logical_and(noise_bu_x_to_keep, noise_bu_y_to_keep)
        noise_bu_dT12 = dT12[noise_bu_to_keep]
        noise_bu_dT13 = dT13[noise_bu_to_keep]
        noise_bu_X1 = X1[noise_bu_to_keep]
        noise_bu_Y1 = Y1[noise_bu_to_keep]
        noise_bu_X2 = X2[noise_bu_to_keep]
        noise_bu_Y2 = Y1[noise_bu_to_keep]

        #noise_dT13_translation = 5*10**-7
        noise_dT13_translation = -1.42*10**-7-noise_bu_y_up
        noise_bu_dT13 = noise_bu_dT13+noise_dT13_translation

        #plt.scatter(dT12, dT13, s=point_size, color="b")
        #plt.scatter(sig_bloc_dT12, sig_bloc_dT13, s=point_size, color="g")
        #plt.scatter(noise_bu_dT12, noise_bu_dT13, s=point_size, color="r")
        #plt.show()

        noise_bu_dT12_r, noise_bu_dT13_r = rotate(noise_bu_dT12, noise_bu_dT13, rot_mat)

        noise_bu_x_to_keep = np.logical_and(noise_bu_dT12_r > sg_x_left, noise_bu_dT12_r < sg_x_right)
        noise_bu_y_to_keep  = np.logical_and(noise_bu_dT13_r > sg_y_bot, noise_bu_dT13_r < sg_y_up)
        noise_bu_to_keep = np.logical_and(noise_bu_x_to_keep, noise_bu_y_to_keep)
        noise_bu_dT12 = noise_bu_dT12[noise_bu_to_keep]
        noise_bu_dT13 = noise_bu_dT13[noise_bu_to_keep]
        noise_bu_X1 = noise_bu_X1[noise_bu_to_keep]
        noise_bu_Y1 = noise_bu_Y1[noise_bu_to_keep]
        noise_bu_X2 = noise_bu_X2[noise_bu_to_keep]
        noise_bu_Y2 = noise_bu_Y2[noise_bu_to_keep]

        noise = (noise_bu_X1, noise_bu_Y1, noise_bu_X2, noise_bu_Y2, noise_bu_dT12,\
        noise_bu_dT13)

        noises_bu.append(noise)

        noise_bu_y_bot = noise_bu_y_bot+delta_y_up

    #noise_bu_y_bot = -1*10**-6
    #noise_bu_y_up = -6*10**-7
    #noise_bu_y_bot =
    #noise_bu_y_up = noise_bu_y_bot+4.5*10**-7
    #
    # noise_bu_x_to_keep = np.logical_and(dT12 > noise_bu_x_left, dT12 < noise_bu_x_right)
    # noise_bu_y_to_keep  = np.logical_and(dT13 > noise_bu_y_bot, dT13 < noise_bu_y_up)
    # noise_bu_to_keep = np.logical_and(noise_bu_x_to_keep, noise_bu_y_to_keep)
    # noise_bu_dT12 = dT12[noise_bu_to_keep]
    # noise_bu_dT13 = dT13[noise_bu_to_keep]
    # noise_bu_X1 = X1[noise_bu_to_keep]
    # noise_bu_Y1 = Y1[noise_bu_to_keep]
    # noise_bu_X2 = X2[noise_bu_to_keep]
    # noise_bu_Y2 = Y1[noise_bu_to_keep]
    #
    # plt.scatter(dT12, dT13, s=point_size, color="b")
    # plt.scatter(sig_bloc_dT12, sig_bloc_dT13, s=point_size, color="g")
    # plt.scatter(noise_bu_dT12, noise_bu_dT13, s=point_size, color="r")
    # plt.show()
    #
    #
    # noise_dT13_translation = 5*10**-7
    # noise_bu_dT13 = noise_bu_dT13+noise_dT13_translation
    #
    # noise_bu_dT12_r, noise_bu_dT13_r = rotate(noise_bu_dT12, noise_bu_dT13, rot_mat)
    #
    # noise_bu_x_to_keep = np.logical_and(noise_bu_dT12_r > sg_x_left, noise_bu_dT12_r < sg_x_right)
    # noise_bu_y_to_keep  = np.logical_and(noise_bu_dT13_r > sg_y_bot, noise_bu_dT13_r < sg_y_up)
    # noise_bu_to_keep = np.logical_and(noise_bu_x_to_keep, noise_bu_y_to_keep)
    # noise_bu_dT12 = noise_bu_dT12[noise_bu_to_keep]
    # noise_bu_dT13 = noise_bu_dT13[noise_bu_to_keep]
    # noise_bu_X1 = noise_bu_X1[noise_bu_to_keep]
    # noise_bu_Y1 = noise_bu_Y1[noise_bu_to_keep]
    # noise_bu_X2 = noise_bu_X2[noise_bu_to_keep]
    # noise_bu_Y2 = noise_bu_Y2[noise_bu_to_keep]
    #
    # noise = (noise_bu_X1, noise_bu_Y1, noise_bu_X2, noise_bu_Y2, noise_bu_dT12,\
    # noise_bu_dT13)
    #
    #
    # plt.scatter(dT12, dT13, s=point_size, color="b")
    # plt.scatter(sig_bloc_dT12, sig_bloc_dT13, s=point_size, color="g")
    # plt.scatter(noise_bu_dT12, noise_bu_dT13, s=point_size, color="r")
    # plt.show()

    # for noise in noises_bu:
    #     noise_bu_dT12 = noise[-2]
    #     noise_bu_dT13 = noise[-1]
    #     plt.scatter(dT12, dT13, s=point_size, color="b")
    #     plt.scatter(sig_bloc_dT12, sig_bloc_dT13, s=point_size, color="g")
    #     plt.scatter(noise_bu_dT12, noise_bu_dT13, s=point_size, color="r")
    #     plt.show()


    return (signal_plus_noise, noises_bu)

def run():

    #From experimental measurements we know:
    #L1 = Z1, L2 = Z2, L3 = Z3: position of detectors 1,2,3 along Z axis
    #Detectors measures positions of particles 1 and 2: X1, Y1, X2, Y2
    #and differences in arrival times: dT12 = t2-1 and dT13 = t3-t1
    #Valocity of center of mass in laboratory frame: \vec{V_cm}

    #Parse acceptance from file
    acceptance_f_path = "../detector_acceptance_3_bodies/acceptance.txt"
    acceptance_f = open(acceptance_f_path, "r")
    acceptance_f_str = acceptance_f.readlines()
    acceptance_str = acceptance_f_str[-1]
    acceptance_kers =  np.linspace(0.01, 6, 500)
    acceptance = np.zeros(acceptance_kers.size)
    number_str = ""
    i = 0
    for char in acceptance_str:
        if char == ",":
            acceptance[i] = float(number_str)
            number_str = ""
            i = i+1
        elif char != " " and char != "[":
            number_str = number_str+char

    def chi_squared(x_v, shift_x, scale_x, scale_y):
        k = 3
        res = np.zeros(x_v.size)

        for i in range(0, x_v.size):
            x = scale_x*x_v[i]+shift_x
            if x <= 0:
                res[i] = 0
            else:
                half_k = k/2.0
                num = x**(half_k-1)*math.exp(-x/2.0)
                den = 2**half_k*sp.special.gamma(half_k)
                res[i] = scale_y*num/den
        return res

    (params, err) = sp.optimize.curve_fit(chi_squared, acceptance_kers, acceptance)
    shift_x, scale_x, scale_y = params
    acceptance_f = lambda x_v: chi_squared(x_v, shift_x, scale_x, scale_y)

    # plt.rc("text", usetex=True)
    # p1, = plt.plot(acceptance_kers, acceptance, label="Simulation")
    # p2, = plt.plot(acceptance_kers,acceptance_f(acceptance_kers), label="$\chi^2_{k=3} fit$")
    # plt.legend(handles=[p1, p2])
    # plt.xlabel("Kinetic energy released (eV)")
    # plt.ylabel("Acceptance")
    #plt.show()


    USE_SIMU_DATA = False

    if USE_SIMU_DATA:
        simu_res_path = "sim_results.txt"
        file = open(simu_res_path, "r")
        file.readline()
        V_cm_line = file.readline()
        V_cm = np.fromstring(V_cm_line, sep=" ")
        print("V_cm "+str(V_cm))
        file.close()
        simu_res = np.loadtxt(simu_res_path, skiprows=3)

        X1 = simu_res[:,0]
        Y1 = simu_res[:,1]
        X2 = simu_res[:,2]
        Y2 = simu_res[:,3]
        X3_simu = simu_res[:,4]
        Y3_simu = simu_res[:,5]
        dT12 = simu_res[:,6]
        dT13 = simu_res[:,7]
        t1_simu = simu_res[:,8]

    else:

        #En colonnes x1, y1, x2, y2, t2-t1, t3-t1
        exp_res_path = "../detector_acceptance_3_bodies/20180622_3body.txt"
        exp_res = np.loadtxt(exp_res_path)
        X1 = exp_res[:,0]*10**-1
        Y1 = exp_res[:,1]*10**-1
        X2 = exp_res[:,2]*10**-1
        Y2 = exp_res[:,3]*10**-1
        dT12 = exp_res[:,4]*10**-9
        dT13 = -exp_res[:,5]*10**-9
        #XU: jeter les évènements pour lesquels x2 est inférieur à 12
        events_to_keep = X2 >= 12*10**-1
        X1 = X1[events_to_keep]
        Y1 = Y1[events_to_keep]
        X2 = X2[events_to_keep]
        Y2 = Y2[events_to_keep]
        dT12 = dT12[events_to_keep]
        dT13 = dT13[events_to_keep]
        BEAM_ENERGY = 18000 #eV
        #Velocity of center of mass in laboratory frame
        V_cm = np.sqrt(BEAM_ENERGY/3)*speed_to_SI_cm*np.array([0,0,1])
        #print("V_CMMMMMMM "+str(V_cm))

    #Find missing info: t1, X3 and Y3

    V_cm_norm = np.linalg.norm(V_cm)


    #
    # plt.hist(X1, 300)
    # plt.title("X1")
    # plt.show()
    #
    # plt.hist(X2, 300)
    # plt.title("X2")
    # plt.show()
    #
    # plt.hist(Y1, 300)
    # plt.title("Y1")
    # plt.show()
    #
    # plt.hist(Y2, 300)
    # plt.title("Y2")
    # plt.show()
    #
    # plt.hist(dT12, 300)
    # plt.title("dT12")
    # plt.show()
    # plt.hist(dT13, 300)
    # plt.title("dT13")
    # plt.show()
    # hist, bins = np.histogram(dT12, np.linspace(np.min(dT12), np.max(dT12), 300))
    # hist = np.abs(fft(hist))
    # width = 0.7 * (bins[1] - bins[0])
    # center = (bins[:-1] + bins[1:]) / 2
    # # plt.bar(center, hist, align='center', width=width, color='g')
    # # plt.show()
    #
    nbr_bars = 200
    #
    # noise_dT12_bool1 = dT12 < -4*10**-8
    # noise_dT12_bool2 = dT12 > -1.5*10**-7
    # noise_dT12_bool = np.logical_and(noise_dT12_bool1, noise_dT12_bool2)
    # noise_dT12 = dT12[noise_dT12_bool]
    # noise_X1 = X1[noise_dT12_bool]
    # noise_Y1 = Y1[noise_dT12_bool]
    # noise_X2 = X2[noise_dT12_bool]
    # noise_Y2 = Y2[noise_dT12_bool]
    # noise_dT13 = dT13[noise_dT12_bool]
    #
    # sorted_by_dT12_idx = np.argsort(noise_dT12)
    # noise_dT12 = noise_dT12[sorted_by_dT12_idx]
    # noise_X1 = X1[sorted_by_dT12_idx]
    # noise_Y1 = Y1[sorted_by_dT12_idx]
    # noise_X2 = X2[sorted_by_dT12_idx]
    # noise_Y2 = Y2[sorted_by_dT12_idx]
    # noise_dT13 = dT13[sorted_by_dT12_idx]
    #
    # # plt.hist(noise_dT12, 300)
    # # plt.title("noise_dT12")
    # # plt.show()
    # noise_dT12_hist, bins = np.histogram(noise_dT12, np.linspace(np.min(dT12), np.max(dT12), nbr_bars))
    # noise_dT12_hist_not_0 = noise_dT12_hist[noise_dT12_hist > 0]



    #
    #
    #
    #
    #exp_data_filt = remove_noisy_events(X1, X2, Y1, Y2, dT12, dT13)
    #
    # #print("events number after remove_noisy_events: "+str(exp_data_filt.shape[0]))
    #
    # # X1r = exp_data_filt[:,0]
    # # Y1r = exp_data_filt[:,1]
    # # X2r = exp_data_filt[:,2]
    # # Y2r = exp_data_filt[:,3]
    #dT12r = exp_data_filt[:,4]
    #dT13r = exp_data_filt[:,5]
    # #
    # #(X1vr,Y1vr,X2vr,Y2vr,X3vr,Y3vr,dT12vr,dT13vr,t1vr) = \
    # #keep_valid_events(X1r, Y1r, X2r, Y2r, dT12r, dT13r,V_cm)
    # #(V1, V2, V3) = compute_v_lab(t1vr, X1vr, Y1vr, X2vr, Y2vr, dT12vr, X3vr, Y3vr, dT13vr)
    # #ker_list_r = compute_ker(V1, V2, V3, V_cm)
    # #hist_r, bins_r = np.histogram(ker_list_r, np.linspace(0, 60, nbr_bars))
    #
    signal_plus_noise, noises = noise_dT12_dT13(X1, Y1, X2, Y2, dT12, dT13)
    X1, Y1, X2, Y2, dT12, dT13 = signal_plus_noise
    val_events = keep_valid_events(X1, Y1, X2, Y2, dT12, dT13, V_cm)
    (X1v,Y1v,X2v,Y2v,X3v,Y3v,dT12v,dT13v,t1v) = val_events
    (V1, V2, V3) = compute_v_lab(t1v, X1v, Y1v, X2v, Y2v, dT12v, X3v, Y3v, dT13v)

    # #t2 = t1v+dT12v
    # #t3 = t1v+dT13v
    #
    # #for i in range(0,len(X1v)):
    # #    print(str(X1v[i])+" "+str(Y1v[i])+" "+str(X2v[i])+" "+str(Y2v[i])+" "+ \
    # #    str(X3v[i])+" "+str(Y3v[i])+" "+str(t1v[i])+" "+str(t2[i])+" "+str(t3[i]))
    #
    # #plot_impacts(V1, V2, V3)
    #
    (total_ker, kerp1, kerp2, kerp3) = compute_ker(V1, V2, V3, V_cm)
    #
    # #ker = np.array(ker_list)
    # print(len(total_ker))
    # total_ker = total_ker[total_ker < 5.2]
    # plt.hist(total_ker, 300)
    # plt.title("ker")
    # plt.show()
    spb, bins = np.histogram(total_ker, np.linspace(0, 6, nbr_bars))
    centers = (bins[:-1] + bins[1:])/2
    spb_accept = acceptance_correction(centers, spb, acceptance_kers, acceptance,
    acceptance_f)

    p1, = plt.plot(centers, spb, label="S+B without acceptance")
    p2, = plt.plot(centers, spb_accept, label="S+B with acceptance")
    plt.legend(handles=[p1, p2])
    plt.show()


    #
    # kerp1 = kerp1[kerp1 < 10]
    # plt.hist(kerp1, 300)
    # plt.title("ker p1")
    # plt.show()
    # hist, bins = np.histogram(kerp1, np.linspace(0, 10, nbr_bars))
    #
    # kerp2 = kerp2[kerp2 < 10]
    # plt.hist(kerp2, 300)
    # plt.title("ker p2")
    # plt.show()
    # hist, bins = np.histogram(kerp2, np.linspace(0, 10, nbr_bars))
    #
    # kerp3 = kerp3[kerp3 < 10]
    # plt.hist(kerp3, 300)
    # plt.title("ker p3")
    # plt.show()
    # hist, bins = np.histogram(kerp3, np.linspace(0, 10, nbr_bars))
    #
    #

    b_tot = np.zeros(nbr_bars-1)
    for noise in noises:
        n_X1, n_Y1, n_X2, n_Y2, n_dT12, n_dT13 = noise
        (X1vs1,Y1vs1,X2vs1,Y2vs1,X3vs1,Y3vs1,dT12vs1,dT13vs1,t1vs1) = \
        keep_valid_events(n_X1, n_Y1, n_X2, n_Y2, n_dT12, n_dT13, V_cm)
        (V1, V2, V3) = compute_v_lab(t1vs1, X1vs1, Y1vs1, X2vs1, Y2vs1, dT12vs1, X3vs1, Y3vs1, dT13vs1)
        (ker_list_noise, n_kerp1, n_kerp2, n_kerp3) = compute_ker(V1, V2, V3, V_cm)
        #ker_hist_noise = np.histogram(ker_list_noise, np.linspace(0, 60, nbr_bars))
        # # ker_list_noise = ker_of_gen_noise(X1, X2, Y1, Y2, dT12, dT13, V_cm_norm)
        b, bins = np.histogram(ker_list_noise, np.linspace(0, 6, nbr_bars))
        b_tot = b_tot+b
    b_tot = b_tot/len(noises)
    centers = (bins[:-1] + bins[1:])/2

    b_accept = acceptance_correction(centers, b, acceptance_kers, acceptance,
    acceptance_f)

    p1, = plt.plot(centers, b, label="B without acceptance")
    p2, = plt.plot(centers, b_accept, label="B with acceptance")
    plt.legend(handles=[p1, p2])
    plt.show()

    # # width = 0.7 * (bins[1] - bins[0])
    # # center = (bins[:-1] + bins[1:]) / 2
    # # plt.bar(center, hist_noise, align='center', width=width, color='g')
    # # #plt.bar(center, hist_mod2, align='center', width=width, color='b')
    # # plt.title("ker_hist of noise")
    # # plt.show()
    # #
    s = spb-b


    centers = (bins[:-1] + bins[1:]) / 2
    width = 0.7 * (bins[1] - bins[0])
    s_accept = acceptance_correction(centers, s, acceptance_kers, acceptance,
    acceptance_f)

    p1, = plt.plot(centers, b_accept, label="B with acceptance")
    p2, = plt.plot(centers, spb_accept, label="S+B with acceptance")
    p3, = plt.plot(centers, s_accept, color="k", linewidth=2,label="S+B-B with acceptance")
    plt.legend(handles=[p1, p2, p3])
    plt.show()

    # #
    # # #plt.plot(center, hist, color='g')
    # # plt.plot(center, ker_hist_mod, color='b')
    # # plt.plot(center, hist_r, color='r')
    # # #plt.plot(center, hist_noise, color='k')
    # # plt.title("ker_hist_mod")
    # # plt.show()
    #

#run()
