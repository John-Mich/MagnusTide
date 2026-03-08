import pyvista as pv
import numpy as np
import time
import sys
import os
from datetime import datetime
import subprocess
import json
import multiprocessing

try:
    import pyi_splash
    # Εδώ μπορείς να βάλεις και κείμενο να αλλάζει (π.χ. "Φόρτωση 3D μηχανής...")
    pyi_splash.update_text('UI Loaded ...')
    
    # Κλείνει την εικόνα για να ανοίξει το κανονικό παράθυρο
    pyi_splash.close()
except ImportError:
    # Αν τρέχεις το .py κανονικά (όχι το .exe), απλά το αγνοεί και προχωράει
    pass

# --- 0. CONFIG & MATH ENGINE ---
pv.global_theme.allow_empty_mesh = True
pv.global_theme.font.color = 'black'

def get_align_matrix(p0, p1, scale_x=1.0, scale_y=1.0, scale_z=1.0):
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
    v = p1 - p0; mag = np.linalg.norm(v)
    if mag < 1e-6 or np.isnan(mag): mag = 0.001; v = np.array([0,0,1], dtype=float)
    else: v = v / mag
    up = np.array([0, 0, 1], dtype=float)
    if np.abs(np.dot(v, up)) > 0.99: up = np.array([0, 1, 0], dtype=float)
    vec_x = np.cross(v, up); vec_x /= np.linalg.norm(vec_x)
    vec_y = np.cross(v, vec_x)
    m = np.eye(4)
    m[0:3, 0] = vec_x * scale_x; m[0:3, 1] = vec_y * scale_y; m[0:3, 2] = v * (mag * scale_z)
    m[0:3, 3] = (p0 + p1) / 2.0
    return m

def get_arrow_align_matrix(p0, p1):
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
    v = p1 - p0; mag = np.linalg.norm(v)
    if mag < 1e-6: mag = 0.001; v = np.array([0,0,1], dtype=float)
    else: v = v / mag
    up = np.array([0, 0, 1], dtype=float)
    if np.abs(np.dot(v, up)) > 0.99: up = np.array([0, 1, 0], dtype=float)
    vec_x = np.cross(v, up); vec_x /= np.linalg.norm(vec_x)
    vec_y = np.cross(v, vec_x)
    m = np.eye(4)
    m[0:3, 0] = vec_x; m[0:3, 1] = vec_y; m[0:3, 2] = v * mag  
    m[0:3, 3] = p0        
    return m

def math_pts_cyl(p0, p1, r0, r1, res=30, rot=0.0):
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
    v = p1 - p0; mag = np.linalg.norm(v)
    if mag < 1e-6: mag = 0.001; v = np.array([0,0,1], dtype=float)
    else: v = v / mag
    not_v = np.array([0, 0, 1], dtype=float)
    if np.abs(np.dot(v, not_v)) > 0.99: not_v = np.array([0, 1, 0], dtype=float)
    n1 = np.cross(v, not_v); n1 /= np.linalg.norm(n1); n2 = np.cross(v, n1)
    num_h = 2; h = np.linspace(0, mag, num_h); u = np.linspace(0, 2*np.pi, res) + rot
    H, U = np.meshgrid(h, u, indexing='ij')
    Radii = np.linspace(r0, r1, num_h)[:, np.newaxis]
    H_ = H[..., np.newaxis]; U_ = U[..., np.newaxis]; R_ = Radii[..., np.newaxis]
    Points = (p0 + v * H_ + R_ * np.cos(U_) * n1 + R_ * np.sin(U_) * n2)
    return Points.reshape(-1, 3), [res, num_h, 1]

def math_pts_strip(p0, p1, r0, r1, angle_c, rot_phase):
    p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float); v = p1 - p0
    mag = np.linalg.norm(v)
    if mag < 1e-6: mag = 0.001; v = np.array([0,0,1], dtype=float)
    else: v = v / mag
    not_v = np.array([0, 0, 1], dtype=float)
    if np.abs(np.dot(v, not_v)) > 0.99: not_v = np.array([0, 1, 0], dtype=float)
    n1 = np.cross(v, not_v); n1 /= np.linalg.norm(n1); n2 = np.cross(v, n1)
    strip_width = np.radians(15.0); res_w = 12 
    u_strip = np.linspace(angle_c - strip_width, angle_c + strip_width, res_w) + rot_phase
    h_grid = np.linspace(0, mag, 2)
    H, U = np.meshgrid(h_grid, u_strip, indexing='ij')
    base_r = r0 + (r1 - r0) * (H / mag)
    Radii = base_r * 1.05 
    H_ = H[..., np.newaxis]; U_ = U[..., np.newaxis]; R_ = Radii[..., np.newaxis]
    Points = (p0 + v * H_ + R_ * np.cos(U_) * n1 + R_ * np.sin(U_) * n2)
    return Points.reshape(-1, 3), [res_w, 2, 1]

def create_solid_template(radius, height=1.0, capping=True):
    return pv.Cylinder(center=(0,0,0), direction=(0,0,1), radius=radius, height=height, resolution=48, capping=capping)
def create_arrow_template(scale=1.0, tip_r=0.1, shaft_r=0.04):
    return pv.Arrow(start=(0,0,0), direction=(0,0,1), scale=scale, tip_length=0.25, tip_radius=tip_r, shaft_radius=shaft_r)
def create_grid_mesh(p0, p1, r0, r1, res=30):
    pts, dims = math_pts_cyl(p0, p1, r0, r1, res, 0.0)
    grid = pv.StructuredGrid(); grid.points = pts; grid.dimensions = dims
    return grid
def create_strip_mesh_init(p0, p1, r0, r1, angle):
    pts, dims = math_pts_strip(p0, p1, r0, r1, angle, 0.0)
    grid = pv.StructuredGrid(); grid.points = pts; grid.dimensions = dims
    return grid

class ScenePart:
    def __init__(self, plotter, mesh, color, opacity=1.0, wireframe=False):
        self.actor = plotter.add_mesh(mesh, color=color, opacity=opacity, style='wireframe' if wireframe else 'surface', 
                                      smooth_shading=True, specular=0.5)
        self.mesh = self.actor.mapper.dataset
        self.user_visible = True 
        
    def set_matrix(self, matrix): self.actor.user_matrix = matrix
    
    def update_transform(self, p0, p1, scale_x=1.0, scale_y=1.0, scale_z=1.0):
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)) or np.isnan(scale_z): return
        self.actor.user_matrix = get_align_matrix(p0, p1, scale_x=scale_x, scale_y=scale_y, scale_z=scale_z)
        
    def update_arrow(self, p0, p1, max_len=30.0):
        p0 = np.array(p0, dtype=float); p1 = np.array(p1, dtype=float)
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)) or np.any(np.isinf(p1)):
            self.actor.visibility = False
            return
            
        v = p1 - p0; mag = np.linalg.norm(v)
        if mag < 0.05 or np.isnan(mag):
            self.actor.visibility = False; return
            
        if mag > max_len: p1 = p0 + (v / mag) * max_len
        self.actor.user_matrix = get_arrow_align_matrix(p0, p1)
        if self.user_visible: self.actor.visibility = True
            
    def set_visibility(self, visible): 
        self.user_visible = bool(visible); self.actor.visibility = self.user_visible

# --- MAIN APP ---
class TwinMagnusHAWT_Physics:
    def __init__(self):
        pv.global_theme.allow_empty_mesh = True
        self.p = pv.Plotter(title="River-Monster: AEROELASTIC V.I.V. META-BEM", window_size=(1600, 1000))
        self.p.set_background('white')
        
        def _on_close(*args):
            try: self.p.iren.TerminateApp()
            except: pass
            sys.exit(0)
        if hasattr(self.p, 'iren') and self.p.iren is not None:
            self.p.iren.add_observer("ExitEvent", _on_close)
            self.p.iren.add_observer("WindowCloseEvent", _on_close)

        self.lut_sr_base = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
        self.lut_cl_base = np.array([0.0, 1.1, 2.6, 4.1, 5.2, 6.0, 6.6, 7.5])
        self.lut_cd_base = np.array([1.1, 0.9, 0.8, 1.1, 1.5, 2.0, 2.6, 3.8])

        self.lut_sr_flap = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0])
        self.lut_cl_flap = np.array([0.0, 1.6, 3.5, 5.2, 6.5, 7.5, 8.2, 9.1])
        self.lut_cd_flap = np.array([1.4, 1.2, 1.1, 1.3, 1.6, 2.0, 2.5, 3.5])

        self.lut_ct = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5])
        self.lut_a  = np.array([0.0, 0.05, 0.11, 0.18, 0.27, 0.38, 0.50, 0.65, 0.82, 1.0, 1.0, 1.0])
            
        self.lut_load_pct = np.array([0.0, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0, 1.2])
        self.lut_eff_gen  = np.array([0.0, 0.35, 0.60, 0.85, 0.92, 0.94, 0.90, 0.82])
        self.lut_eff_mot  = np.array([0.0, 0.30, 0.55, 0.80, 0.88, 0.90, 0.86, 0.75])

        self.rho = 1000.0 
        self.Hub_R = 4.0 
        self.sim_time = 0.0 
        
        self.val_cyl_L = 8.0
        self.val_cyl_r = 0.3 
        self.val_rib_frac = 0.0        
        self.val_rib_cl_max = 3.5      
        self.val_rib_sr_peak = 3.0    
        
        self.Cyl_L = self.val_cyl_L
        self.Cyl_r = self.val_cyl_r
        self.wall_thickness = 0.015  
        self.r_in = self.Cyl_r - self.wall_thickness
        
        self.Bridge_Mass = 6000.0 
        self.System_Inertia_Pitch = 1200000.0 
        self.System_Inertia_Yaw = 1800000.0 
        
        self.update_mass_properties()
        
        self.sys_pos = np.array([0.0, 0.0, -1.0]) 
        self.sys_vel = np.array([0.0, 0.0, 0.0])
        self.sys_pitch = 0.0; self.sys_omega_pitch = 0.0
        self.sys_yaw = 0.0; self.sys_omega_yaw = 0.0 
        
        self.anchor_pos_front = np.array([0.0, -60.0, -45.0])
        self.anchor_pos_rear = np.array([0.0, 60.0, -45.0])
        self.Hub_Dist_Mean = 24.0   
        self.Frame_Depth = 12.0
        self.toe_in_angle = 0.0
        
        Pontoon_Vol_Effective = 170.0 
        self.Buoyancy_Frame_Total_N = Pontoon_Vol_Effective * 1000.0 * 9.81
        
        self.spinning = False; self.was_spinning = False 
        self.is_paused = False # TIME FREEZE flag
        
        self.lut_v =     [1.0,  1.17, 1.52, 1.81, 2.09, 2.61, 4.40]
        self.lut_vsurf = [2.81, 2.81, 2.81, 2.81, 3.25, 4.11, 4.55]
        self.lut_load =  [4.5,  6.0,  7.4,  8.9,  19.0, 37.0, 57.9]
        self.ap_prev_water = 0.0
        
        self.autopilot_on = False; self.ap_timer = 0.0; self.ap_interval = 2.5 
        self.ap_phase = 'SPIN'; self.ap_dir_spin = 1; self.ap_dir_load = 1
        self.ap_step_vsurf = 0.2; self.ap_step_load = 1.0; self.ap_prev_net = -999999.0
        
        self.show_flow_vectors = False
        self.show_force_total = False; self.show_force_components = True
        self.show_stress_forces = False
        self.show_flaps = False 
        self.freeze_base = False 
        
        self.val_water_speed = 1.8   
        self.val_main_line = 50.0  
        self.val_winch_pitch = 0.0  
        self.val_winch_yaw = 0.0    
        
        self.val_vsurf_L = 4.0; self.val_vsurf_R = -4.0   
        self.val_gen_load = 25.0      
        
        self.omega_L = 0.0; self.omega_R = 0.0 
        self.rotor_angle_L = 0.0; self.rotor_angle_R = 0.0
        self.cyl_spins = {i: 0.0 for i in range(8)}
        self.cyl_phases = {i: 0.0 for i in range(8)}
        
        self.tension_FL = 0.0; self.tension_FR = 0.0; self.tension_RL = 0.0; self.tension_RR = 0.0
        self.tension_Main_Front = 0.0; self.tension_Main_Rear = 0.0
        self.beam_tension_X = 0.0; self.beam_torsion_Z = 0.0
        
        self.SR_L = 0.0; self.CL_L = 0.0; self.CD_L = 0.0; self.a_L = 0.0
        self.SR_R = 0.0; self.CL_R = 0.0; self.CD_R = 0.0; self.a_R = 0.0
        self.thrust_L = 0.0; self.thrust_R = 0.0; self.drive_L = 0.0; self.drive_R = 0.0
        
        self.P_mot_L = 0.0; self.P_mot_R = 0.0
        self.P_gen_mech_L = 0.0; self.P_gen_mech_R = 0.0
        self.P_gen_elec_L = 0.0; self.P_gen_elec_R = 0.0
        self.net_P_W = 0.0; self.total_P_gen_W = 0.0; self.total_P_mot_W = 0.0
        self.mech_P_gen_W = 0.0; self.hydro_power_in_W = 0.0
        self.avg_flap_penalty = 0.0
        
        self.eff_mot_L = 0.0; self.eff_mot_R = 0.0
        self.eff_gen_L = 0.0; self.eff_gen_R = 0.0

        self.frame_parts = {}; self.env_parts = {}; self.rotor_parts = []; self.dynamic_blades = []
        self.blade_vector_parts = {}; self.lbl_actors = [] 
        
        self.setup_ui(); self.build_scene(); self.update_hud()
        self.p.camera.position = (45, -100, 45); self.p.camera.focal_point = (0, 0, -10); self.p.camera.up = (0, 0, 1)

    def update_mass_properties(self):
        Volume_PVC = np.pi * (self.Cyl_r**2 - self.r_in**2) * self.Cyl_L
        Structural_Mass = Volume_PVC * 1380.0
        Mass_Water_Inside = np.pi * (self.r_in**2) * self.Cyl_L * self.rho
        self.Cyl_Mass = Structural_Mass + Mass_Water_Inside  
        self.I_cyl = 0.5 * self.Cyl_Mass * (self.Cyl_r ** 2) 
        self.Cyl_Mid_R = self.Hub_R + (self.Cyl_L / 2.0) 
        
        self.System_Mass = 160000.0  
        self.Inertia_Rotor = 3.0 * (self.Cyl_Mass * (self.Cyl_Mid_R**2)) + 10000.0

    def draw_button_labels(self):
        c1, c2, c3, c4 = 50, 350, 700, 1050
        r1, r2, r3 = 120, 75, 30
        
        self.lbl_actors = [
            self.p.add_text("LOAD CFD STATE", position=(c1 + 40, r2 + 5), color='orange', font_size=12),
            self.p.add_text("EXPORT 3D CFD CASE", position=(c1 + 40, r3 + 5), color='green', font_size=12),
            
            self.p.add_text("START SIM", position=(c2 + 40, r1 + 5), color='black', font_size=12),
            self.p.add_text("TIME FREEZE (PAUSE)", position=(c2 + 40, r2 + 5), color='red', font_size=12),
            self.p.add_text("AP 3.0 (TUNE & PITCH)", position=(c2 + 40, r3 + 5), color='red', font_size=12),
            
            self.p.add_text("TOTAL FORCE", position=(c3 + 40, r1 + 5), color='purple', font_size=12),
            self.p.add_text("FORCE COMPONENTS", position=(c3 + 40, r2 + 5), color='blue', font_size=12),
            self.p.add_text("STRESS VECTORS", position=(c3 + 40, r3 + 5), color='red', font_size=12),
            
            self.p.add_text("MAGNUS FLOW", position=(c4 + 40, r1 + 5), color='cyan', font_size=12),
            self.p.add_text("ARTICULATED FLAPS ON", position=(c4 + 40, r2 + 5), color='orange', font_size=12),
            self.p.add_text("FREEZE BASE SHAPE", position=(c4 + 40, r3 + 5), color='purple', font_size=12)
        ]

    def setup_ui(self):
        self.slider_water = self.p.add_slider_widget(self.set_w, [0, 8.0], title="Water Speed (m/s)", value=self.val_water_speed, pointa=(0.01, 0.93), pointb=(0.14, 0.93), style='modern')
        self.slider_main = self.p.add_slider_widget(self.set_m, [10, 100], title="Main Lines (m)", value=self.val_main_line, pointa=(0.01, 0.81), pointb=(0.14, 0.81), style='modern')
        self.slider_pitch = self.p.add_slider_widget(self.set_wp, [-10.0, 10.0], title="Vertical Pitch (m)", value=self.val_winch_pitch, pointa=(0.01, 0.69), pointb=(0.14, 0.69), style='modern')
        self.slider_yaw = self.p.add_slider_widget(self.set_wy, [-10.0, 10.0], title="Horiz Yaw (m)", value=self.val_winch_yaw, pointa=(0.01, 0.57), pointb=(0.14, 0.57), style='modern')
        self.slider_spin_L = self.p.add_slider_widget(self.set_sl, [-15.0, 15.0], title="Left V_surf (m/s)", value=self.val_vsurf_L, pointa=(0.01, 0.45), pointb=(0.14, 0.45), style='modern')
        self.slider_spin_R = self.p.add_slider_widget(self.set_sr, [-15.0, 15.0], title="Right V_surf (m/s)", value=self.val_vsurf_R, pointa=(0.01, 0.33), pointb=(0.14, 0.33), style='modern')
        self.slider_load = self.p.add_slider_widget(self.set_ld, [0, 100], title="Gen Load (%)", value=self.val_gen_load, pointa=(0.01, 0.21), pointb=(0.14, 0.21), style='modern')
        
        self.slider_cyl_L = self.p.add_slider_widget(self.set_cl, [4.0, 16.0], title="Cyl Length (m)", value=self.val_cyl_L, pointa=(0.18, 0.93), pointb=(0.31, 0.93), style='modern')
        self.slider_cyl_r = self.p.add_slider_widget(self.set_cr, [0.1, 0.5], title="Cyl Radius (m)", value=self.val_cyl_r, pointa=(0.18, 0.81), pointb=(0.31, 0.81), style='modern')
        self.slider_rib_f = self.p.add_slider_widget(self.set_rf, [0.0, 1.0], title="Ribbed Zone (%)", value=self.val_rib_frac, pointa=(0.18, 0.69), pointb=(0.31, 0.69), style='modern')
        self.slider_rib_c = self.p.add_slider_widget(self.set_rc, [2.0, 5.0], title="Ribbed Max CL", value=self.val_rib_cl_max, pointa=(0.18, 0.57), pointb=(0.31, 0.57), style='modern')
        self.slider_rib_s = self.p.add_slider_widget(self.set_rs, [1.5, 4.0], title="Ribbed Peak SR", value=self.val_rib_sr_peak, pointa=(0.18, 0.45), pointb=(0.31, 0.45), style='modern')

        c1, c2, c3, c4 = 50, 350, 700, 1050
        r1, r2, r3 = 120, 75, 30

        self.p.add_checkbox_button_widget(self.load_state_gui, value=False, position=(c1, r2), size=30, color_on='orange', color_off='grey')
        self.p.add_checkbox_button_widget(self.export_cfd_gui, value=False, position=(c1, r3), size=30, color_on='lime', color_off='green')

        self.p.add_checkbox_button_widget(self.toggle_spin, value=self.spinning, position=(c2, r1), size=30, color_on='green', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_pause, value=self.is_paused, position=(c2, r2), size=30, color_on='red', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_autopilot, value=self.autopilot_on, position=(c2, r3), size=30, color_on='red', color_off='grey')

        self.p.add_checkbox_button_widget(self.toggle_force_total, value=self.show_force_total, position=(c3, r1), size=30, color_on='purple', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_force_comp, value=self.show_force_components, position=(c3, r2), size=30, color_on='blue', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_stress, value=self.show_stress_forces, position=(c3, r3), size=30, color_on='red', color_off='grey')

        self.p.add_checkbox_button_widget(self.toggle_flow, value=self.show_flow_vectors, position=(c4, r1), size=30, color_on='cyan', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_flaps, value=self.show_flaps, position=(c4, r2), size=30, color_on='orange', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_freeze, value=self.freeze_base, position=(c4, r3), size=30, color_on='purple', color_off='grey')

        self.draw_button_labels()

    def set_w(self, v): self.val_water_speed = v
    def set_m(self, v): self.val_main_line = v
    def set_wp(self, v): self.val_winch_pitch = v
    def set_wy(self, v): self.val_winch_yaw = v
    def set_sl(self, v): self.val_vsurf_L = v
    def set_sr(self, v): self.val_vsurf_R = v
    def set_ld(self, v): self.val_gen_load = v
    def set_cl(self, v): self.val_cyl_L = v
    def set_cr(self, v): self.val_cyl_r = v
    def set_rf(self, v): self.val_rib_frac = v
    def set_rc(self, v): self.val_rib_cl_max = v
    def set_rs(self, v): self.val_rib_sr_peak = v

    def toggle_spin(self, state): self.spinning = state
    def toggle_autopilot(self, state): self.autopilot_on = state; self.ap_timer = 0.0; self.ap_prev_net = self.net_P_W
    def toggle_force_total(self, state): self.show_force_total = state; self.update_geometry()
    def toggle_force_comp(self, state): self.show_force_components = state; self.update_geometry()
    def toggle_stress(self, state): self.show_stress_forces = state; self.update_geometry()
    def toggle_flow(self, state): self.show_flow_vectors = state; self.update_geometry()
    def toggle_flaps(self, state): self.show_flaps = state; self.update_geometry()
    def toggle_freeze(self, state): self.freeze_base = state; self.update_geometry()
    def toggle_pause(self, state): self.is_paused = state

    def update_hud(self):
        w_spd = self.val_water_speed; dpth = self.sys_pos[2]; ptch = np.degrees(self.sys_pitch); yw = np.degrees(self.sys_yaw)
        fb = self.Buoyancy_Frame_Total_N / 9810.0
        t_mf = self.tension_Main_Front / 1000.0; t_mr = self.tension_Main_Rear / 1000.0
        t_fl = self.tension_FL / 1000.0; t_fr = self.tension_FR / 1000.0
        t_rl = self.tension_RL / 1000.0; t_rr = self.tension_RR / 1000.0; t_beam = self.beam_tension_X / 1000.0
        t_torsion = self.beam_torsion_Z / 1000.0; thr_l = self.thrust_L / 1000.0; thr_r = self.thrust_R / 1000.0
        rpm_hub_l = self.omega_L * 9.54929; rpm_hub_r = self.omega_R * 9.54929
        rpm_cyl_l = self.cyl_spins[0]; rpm_cyl_r = self.cyl_spins[3]
        sr_l = self.SR_L; sr_r = self.SR_R; cl_l, cd_l = self.CL_L, self.CD_L; cl_r, cd_r = self.CL_R, self.CD_R
        drv_l = self.drive_L / 1000.0; drv_r = self.drive_R / 1000.0
        pm_l = self.P_mot_L / 1000.0; pm_r = self.P_mot_R / 1000.0
        pge_l = self.P_gen_elec_L / 1000.0; pge_r = self.P_gen_elec_R / 1000.0
        net = self.net_P_W / 1000.0; hydro_in = self.hydro_power_in_W / 1000.0
        a_L_hud = self.a_L; a_R_hud = self.a_R
        
        eff_m_l = self.eff_mot_L * 100.0; eff_m_r = self.eff_mot_R * 100.0
        eff_g_l = self.eff_gen_L * 100.0; eff_g_r = self.eff_gen_R * 100.0
        
        ap_status = f"ACTIVE (Phase: {self.ap_phase} | Trim: Auto)" if self.autopilot_on else "OFF"
        rib_status = f"{self.val_rib_frac*100:.0f}% Ribbed"
        
        base_status = "LOCKED (Off-Design)" if self.freeze_base else "DYNAMIC (Ideal)"
        flap_status = f"ON | Base: {base_status} | Aero Mismatch Penalty: {self.avg_flap_penalty*100:.1f}%" if self.show_flaps else "OFF"
        
        v_s_l = rpm_cyl_l * 0.1047 * self.Cyl_r
        v_s_r = rpm_cyl_r * 0.1047 * self.Cyl_r

        avg_a = (self.a_L + self.a_R) / 2.0
        if avg_a > 0.45: aero_status = f"!! WARNING: SEVERE BLOCKAGE (a={avg_a:.2f}) - FLOW BYPASS !! "
        elif avg_a > 0.35: aero_status = f"High Blockage (a={avg_a:.2f}) - Approaching Spillage Limits"
        else: aero_status = f"Optimal Flow (a={avg_a:.2f}) - BEM Strictly Valid"

        max_tension_tons = max(self.tension_Main_Front, self.tension_Main_Rear, self.tension_FL, self.tension_FR) / 9810.0
        max_thrust_tons = max(self.thrust_L, self.thrust_R) / 9810.0
        
        if max_tension_tons > 800.0 or max_thrust_tons > 500.0:
            structural_status = "!! DEVICE DESTROYED: STRUCTURAL FAILURE !!"
        elif max_tension_tons > 600.0 or max_thrust_tons > 350.0:
            structural_status = "WARNING: CRITICAL STRESS LIMITS EXCEEDED"
        else:
            structural_status = "NOMINAL (Safe Operating Loads)"

        hud = f"""
=====================================================================================================
 FLIGHT DECK / TELEMETRY OVERVIEW   |   [ AEROELASTIC META-BEM ] 
=====================================================================================================
  AERO DYNAMICS:  {aero_status}
  STRUCTURE    :  {structural_status}
  FLAP SYSTEM  :  {flap_status}
-----------------------------------------------------------------------------------------------------
 [ KINEMATICS & ENVIRONMENT ]             [ STRUCTURAL STRESS & MOORING (kN) ]
  Water Speed  : {w_spd:>6.2f} m/s                  Main Lines F/R : {t_mf:>6.1f} / {t_mr:>6.1f} kN
  Sys Depth    : {dpth:>6.2f} m                         Bridle FL/FR   : {t_fl:>8.1f} / {t_fr:>8.1f} kN
  Sys Pitch    : {ptch:>6.2f} deg                       Bridle RL/RR   : {t_rl:>8.1f} / {t_rr:>8.1f} kN
  Sys Yaw      : {yw:>6.2f} deg                         Rear Beam      : {t_beam:>8.1f} kN (Tension)
  Frame Buoy   : {fb:>6.1f} t                        Beam Torsion   : {t_torsion:>8.1f} kNm
                                          Rotor Thrust   : L:{thr_l:>6.1f} | R:{thr_r:>6.1f} kN

 [ PORT ROTOR (LEFT) TELEMETRY ]          [ STARBOARD ROTOR (RIGHT) TELEMETRY ]
  V_surface    : {v_s_l:>6.2f} m/s ({rpm_cyl_l:>4.0f} RPM)       V_surface    : {v_s_r:>6.2f} m/s ({rpm_cyl_r:>4.0f} RPM)
  Hub Orbit    : {rpm_hub_l:>6.1f} RPM                  Hub Orbit    : {rpm_hub_r:>6.1f} RPM
  Spin Ratio   : {sr_l:>6.2f} (a_vperp)             Spin Ratio   : {sr_r:>6.2f} (a_vperp)
  Aero Polars  : CL={cl_l:>4.2f} | CD={cd_l:>4.2f}         Aero Polars  : CL={cl_r:>4.2f} | CD={cd_r:>4.2f}
  Induction (a): {a_L_hud:>4.2f} (Flow Blockage)        Induction (a): {a_R_hud:>4.2f} (Flow Blockage)
  Drive Force  : {drv_l:>6.1f} kN                  Drive Force  : {drv_r:>6.1f} kN

 [ POWER SYSTEMS & EFFICIENCY ]
  Hydro Power In (Avail) : {hydro_in:>8.2f} kW
  PORT (L)  : Mot Draw: {pm_l:>7.2f} kW (Eff: {eff_m_l:>2.0f}%) | Gen Elec: +{pge_l:>7.2f} kW (Eff: {eff_g_l:>2.0f}%)
  STBD (R)  : Mot Draw: {pm_r:>7.2f} kW (Eff: {eff_m_r:>2.0f}%) | Gen Elec: +{pge_r:>7.2f} kW (Eff: {eff_g_r:>2.0f}%)
-----------------------------------------------------------------------------------------------------
  SYSTEM NET ELECTRICAL POWER :  {net:>+8.2f} kW  |  MPPT: {ap_status}
=====================================================================================================
"""
        width, height = self.p.window_size
        self.p.add_text(hud, position=(width - 780, height - 420), color='black', font_size=7, font='courier', name='hud_block')

    def build_scene(self):
        p_L = np.array([-1, 0, 0]); p_R = np.array([ 1, 0, 0])
        self.frame_parts['Beam_Front'] = ScenePart(self.p, create_grid_mesh(p_L, p_R, 0.2, 0.2), 'grey') 
        self.frame_parts['Beam_Rear']  = ScenePart(self.p, create_grid_mesh(p_L, p_R, 0.2, 0.2), 'grey')
        
        self.frame_parts['Beam_Left']  = ScenePart(self.p, pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30), 'yellow')
        self.frame_parts['Beam_Right'] = ScenePart(self.p, pv.Sphere(radius=1.0, theta_resolution=30, phi_resolution=30), 'yellow')
        
        self.frame_parts['Motor_Bell_L'] = ScenePart(self.p, create_solid_template(1.0, height=1.0), 'darkgrey')
        self.frame_parts['Motor_Bell_R'] = ScenePart(self.p, create_solid_template(1.0, height=1.0), 'darkgrey')
        self.frame_parts['Drive_Shaft_L'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.1, 0.1), 'silver')
        self.frame_parts['Drive_Shaft_R'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.1, 0.1), 'silver')
        
        self.env_parts['Anchor_Front'] = ScenePart(self.p, pv.Box(bounds=(-2, 2, -2, 2, -1, 1)), 'darkgrey')
        self.env_parts['Anchor_Rear']  = ScenePart(self.p, pv.Box(bounds=(-2, 2, -2, 2, -1, 1)), 'darkgrey')
        self.env_parts['Knot_Front'] = ScenePart(self.p, pv.Sphere(radius=0.6), 'red')
        self.env_parts['Knot_Rear']  = ScenePart(self.p, pv.Sphere(radius=0.6), 'red')
        
        self.env_parts['Main_Line_Front'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.1, 0.1), 'black')
        self.env_parts['Main_Line_Rear']  = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.1, 0.1), 'black')
        
        self.env_parts['Bridle_FL'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.04, 0.04), 'black')
        self.env_parts['Bridle_FR'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.04, 0.04), 'black')
        self.env_parts['Bridle_RL'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.04, 0.04), 'black')
        self.env_parts['Bridle_RR'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.04, 0.04), 'black')
        
        for tag in ['Tension_FL', 'Tension_FR', 'Tension_RL', 'Tension_RR', 'Tension_Main_Front', 'Tension_Main_Rear']:
            self.env_parts[tag] = ScenePart(self.p, create_arrow_template(scale=1.5, tip_r=0.15, shaft_r=0.06), 'red')
            self.env_parts[tag].set_visibility(False)

        blade_idx = 0 
        for tag, rotor_id in [('Left', 0), ('Right', 1)]:
            rotor_parts = {}
            rotor_parts['Hub'] = ScenePart(self.p, create_solid_template(self.Hub_R, height=0.6, capping=False), 'silver')
            rotor_parts['Hub_Ext'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.15, 0.15), 'silver') 
            rotor_parts['Pontoon_Rim'] = ScenePart(self.p, create_solid_template(3.05, height=0.2, capping=False), 'darkgrey')
            self.rotor_parts.append(rotor_parts)
            
            for deg in [0, 120, 240]:
                strut = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.1, 0.1), 'black')
                motor_bell_inner = ScenePart(self.p, create_solid_template(self.Cyl_r * 0.8, height=0.5), 'black')
                blade_cyl = ScenePart(self.p, create_grid_mesh([0,0,-self.Cyl_L/2], [0,0,self.Cyl_L/2], self.Cyl_r, self.Cyl_r, res=30), 'blue', opacity=0.8)
                tape = ScenePart(self.p, create_strip_mesh_init([0,0,-self.Cyl_L/2], [0,0,self.Cyl_L/2], self.Cyl_r, self.Cyl_r, 0.0), 'black')
                plate_top = ScenePart(self.p, create_solid_template(self.Cyl_r*2.5, height=0.05), 'darkgrey')
                plate_bot = ScenePart(self.p, create_solid_template(self.Cyl_r*2.5, height=0.05), 'darkgrey')
                wire = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.03, 0.03), 'black') 
                
                rod_mesh = pv.StructuredGrid()
                rod_mesh.points = np.zeros((20, 3))
                rod_mesh.dimensions = [2, 10, 1]
                rod_actor = ScenePart(self.p, rod_mesh, 'darkgrey', opacity=1.0)
                rod_actor.set_visibility(False)
                
                flaps = []
                for _ in range(10):
                    f_mesh = pv.StructuredGrid()
                    f_mesh.points = np.zeros((4, 3))
                    f_mesh.dimensions = [2, 2, 1]
                    f_act = ScenePart(self.p, f_mesh, 'orange', opacity=0.85)
                    f_act.set_visibility(False)
                    flaps.append(f_act)
                
                vectors = {}
                for key, col, scale, t_r, s_r in [
                    ('force_total','purple',1.5, 0.1, 0.04), ('force_drive','blue',1.5, 0.1, 0.04), ('force_axial','red',1.5, 0.1, 0.04),
                    ('stress_tip', 'magenta', 1.5, 0.15, 0.06), ('stress_hub', 'red', 1.5, 0.15, 0.06),
                    ('flow_A', 'cyan', 1.0, 0.08, 0.03), ('flow_B', 'cyan', 1.0, 0.08, 0.03)]:
                    vectors[key] = ScenePart(self.p, create_arrow_template(scale=scale, tip_r=t_r, shaft_r=s_r), col)
                    vectors[key].set_visibility(False)
                
                strips_data = [{'a': 0.0, 'a_prime': 0.0, 'flap_dir': None} for _ in range(10)]
                
                self.dynamic_blades.append({
                    'tag': tag, 'rotor_id': rotor_id, 'base_angle': np.radians(deg), 
                    'strut': strut, 'bell_inner': motor_bell_inner,
                    'plate_top': plate_top, 'plate_bot': plate_bot,
                    'blade_cyl': blade_cyl, 'tape': tape, 'wire': wire, 
                    'rod': rod_actor, 'rod_pts': np.zeros((20, 3)),
                    'flaps': flaps, 'frozen_twist': np.zeros(10), 'current_base_ref': 0.0,
                    'id': blade_idx, 'strips': strips_data
                })
                self.blade_vector_parts[blade_idx] = vectors
                blade_idx += 1
        self.update_geometry()

    def update_geometry(self):
        dt = 0.04 
        self.sim_time += dt 
        
        if self.val_cyl_L != self.Cyl_L or self.val_cyl_r != self.Cyl_r:
            self.Cyl_L = self.val_cyl_L; self.Cyl_r = self.val_cyl_r; self.r_in = self.Cyl_r - self.wall_thickness
            self.update_mass_properties()

        self.P_mot_L = 0.0; self.P_mot_R = 0.0
        self.tension_FL = 0.0; self.tension_FR = 0.0; self.tension_RL = 0.0; self.tension_RR = 0.0
        self.tension_Main_Front = 0.0; self.tension_Main_Rear = 0.0
        total_aero_penalty = 0.0 
        
        cp = np.cos(self.sys_pitch); sp = np.sin(self.sys_pitch)
        cy = np.cos(self.sys_yaw);   sy = np.sin(self.sys_yaw)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        R_mat = Rz.dot(Rx) 
        
        def get_x_offset(y_local): return (self.Hub_Dist_Mean/2) + y_local * np.tan(self.toe_in_angle)
            
        y_front = -self.Frame_Depth/2; y_rear = self.Frame_Depth/2; hub_offset_y = 0.0 
        x_front = get_x_offset(y_front); x_rear = get_x_offset(y_rear); x_hub = get_x_offset(hub_offset_y)
        
        m_FL = self.sys_pos + R_mat.dot(np.array([-x_front, y_front, 0]))
        m_FR = self.sys_pos + R_mat.dot(np.array([ x_front, y_front, 0]))
        m_RL = self.sys_pos + R_mat.dot(np.array([-x_rear,  y_rear, 0]))
        m_RR = self.sys_pos + R_mat.dot(np.array([ x_rear,  y_rear, 0]))
        hub_L_global = self.sys_pos + R_mat.dot(np.array([-x_hub, hub_offset_y, 0]))
        hub_R_global = self.sys_pos + R_mat.dot(np.array([ x_hub, hub_offset_y, 0]))
        mid_L = (m_FL + m_RL) / 2.0; mid_R = (m_FR + m_RR) / 2.0
        
        self.frame_parts['Beam_Front'].mesh.points, _ = math_pts_cyl(m_FL, m_FR, 0.2, 0.2)
        self.frame_parts['Beam_Rear'].mesh.points, _  = math_pts_cyl(m_RL, m_RR, 0.2, 0.2)
        self.frame_parts['Beam_Left'].update_transform(m_FL, m_RL, scale_x=3.0, scale_y=3.0, scale_z=1.0)
        self.frame_parts['Beam_Right'].update_transform(m_FR, m_RR, scale_x=3.0, scale_y=3.0, scale_z=1.0)
        
        rotor_z_L_local = np.array([np.sin(self.toe_in_angle), -np.cos(self.toe_in_angle), 0])
        rotor_x_L_local = np.array([np.cos(self.toe_in_angle),  np.sin(self.toe_in_angle), 0])
        rotor_y_local   = np.array([0, 0, 1])
        rotor_z_R_local = np.array([-np.sin(self.toe_in_angle), -np.cos(self.toe_in_angle), 0])
        rotor_x_R_local = np.array([ np.cos(self.toe_in_angle), -np.sin(self.toe_in_angle), 0])
        rot_axis_L_global = R_mat.dot(rotor_z_L_local); rot_axis_R_global = R_mat.dot(rotor_z_R_local)
        
        for idx, mid_pos in enumerate([mid_L, mid_R]):
            tag = 'L' if idx == 0 else 'R'
            hub_g = hub_L_global if idx == 0 else hub_R_global
            p_pod_top = mid_pos + R_mat.dot(np.array([0, 0, -3.0]))
            p_pod_bot = mid_pos + R_mat.dot(np.array([0, 0, -3.8])) 
            self.frame_parts[f'Motor_Bell_{tag}'].update_transform(p_pod_bot, p_pod_top)
            p_rim_bot = hub_g + R_mat.dot(np.array([0, 0, -self.Hub_R]))
            self.frame_parts[f'Drive_Shaft_{tag}'].mesh.points, _ = math_pts_cyl(p_pod_bot, p_rim_bot, 0.1, 0.1)

        for i, hub_global in enumerate([hub_L_global, hub_R_global]):
            parts = self.rotor_parts[i]
            r_z = rotor_z_L_local if i == 0 else rotor_z_R_local; r_x = rotor_x_L_local if i == 0 else rotor_x_R_local
            r_axis_glob = rot_axis_L_global if i == 0 else rot_axis_R_global
            m_hub = np.eye(4); m_hub[0:3, 0] = R_mat.dot(r_x); m_hub[0:3, 1] = R_mat.dot(rotor_y_local); m_hub[0:3, 2] = R_mat.dot(r_z); m_hub[0:3, 3] = hub_global
            parts['Hub'].set_matrix(m_hub); parts['Pontoon_Rim'].set_matrix(m_hub)
            p_ext = hub_global + r_axis_glob * 2.5
            parts['Hub_Ext'].mesh.points, _ = math_pts_cyl(hub_global, p_ext, 0.15, 0.15)

        A_swept = np.pi * ((self.Hub_R + self.Cyl_L)**2 - self.Hub_R**2) 
        self.hydro_power_in_W = 0.5 * self.rho * (A_swept * 2) * (self.val_water_speed**3)

        v_water_global = np.array([0.0, self.val_water_speed, 0.0])
        Sys_Net_Force = np.array([0.0, 0.0, -self.System_Mass * 9.81]) 
        Sys_Net_Torque = np.array([0.0, 0.0, 0.0]) 
        
        Buoy_per_pontoon = self.Buoyancy_Frame_Total_N / 2.0
        F_buoy = np.array([0.0, 0.0, Buoy_per_pontoon])
        Sys_Net_Force += (2 * F_buoy)
        Sys_Net_Torque += np.cross(mid_L - self.sys_pos, F_buoy); Sys_Net_Torque += np.cross(mid_R - self.sys_pos, F_buoy)
        
        m_anc_f = np.eye(4); m_anc_f[0:3, 3] = self.anchor_pos_front; self.env_parts['Anchor_Front'].set_matrix(m_anc_f)
        m_anc_r = np.eye(4); m_anc_r[0:3, 3] = self.anchor_pos_rear; self.env_parts['Anchor_Rear'].set_matrix(m_anc_r)
        
        vec_anc_f_to_center = self.sys_pos - self.anchor_pos_front; dist_f = np.linalg.norm(vec_anc_f_to_center)
        knot_pos_front = self.anchor_pos_front + (vec_anc_f_to_center / max(dist_f, 0.001)) * self.val_main_line
        vec_anc_r_to_center = self.sys_pos - self.anchor_pos_rear; dist_r = np.linalg.norm(vec_anc_r_to_center)
        knot_pos_rear = self.anchor_pos_rear + (vec_anc_r_to_center / max(dist_r, 0.001)) * self.val_main_line
        
        base_winch_len = 25.0
        w_FL = base_winch_len - self.val_winch_pitch + self.val_winch_yaw
        w_FR = base_winch_len - self.val_winch_pitch - self.val_winch_yaw
        w_RL = base_winch_len + self.val_winch_pitch + self.val_winch_yaw
        w_RR = base_winch_len + self.val_winch_pitch - self.val_winch_yaw
        
        winches = [
            (m_FL, w_FL, 'Bridle_FL', 'Tension_FL', knot_pos_front),
            (m_FR, w_FR, 'Bridle_FR', 'Tension_FR', knot_pos_front),
            (m_RL, w_RL, 'Bridle_RL', 'Tension_RL', knot_pos_rear),
            (m_RR, w_RR, 'Bridle_RR', 'Tension_RR', knot_pos_rear)
        ]
        
        for corner_pos, winch_len, part_name, tension_name, k_pos in winches:
            vec_corner_to_knot = k_pos - corner_pos; dist_ck = np.linalg.norm(vec_corner_to_knot)
            stretch_ck = dist_ck - winch_len
            if stretch_ck > 0:
                F_pull_mag = stretch_ck * 50000.0
                if part_name == 'Bridle_FL': self.tension_FL = F_pull_mag
                elif part_name == 'Bridle_FR': self.tension_FR = F_pull_mag
                elif part_name == 'Bridle_RL': self.tension_RL = F_pull_mag
                elif part_name == 'Bridle_RR': self.tension_RR = F_pull_mag
                
                F_pull_vec = F_pull_mag * (vec_corner_to_knot / max(dist_ck, 0.001))
                Sys_Net_Force += F_pull_vec
                Sys_Net_Torque += np.cross(corner_pos - self.sys_pos, F_pull_vec)
                
                if self.show_stress_forces:
                    self.env_parts[tension_name].set_visibility(True)
                    self.env_parts[tension_name].update_arrow(corner_pos, corner_pos + F_pull_vec / 50000.0)
                else: self.env_parts[tension_name].set_visibility(False)
            else: self.env_parts[tension_name].set_visibility(False)
                
            self.env_parts[part_name].mesh.points, _ = math_pts_cyl(corner_pos, k_pos, 0.04, 0.04)
        
        stretch_anc_f = dist_f - self.val_main_line
        if stretch_anc_f > 0: 
            self.tension_Main_Front = stretch_anc_f * 100000.0
            if self.show_stress_forces:
                dir_f = vec_anc_f_to_center / max(dist_f, 0.001)
                v_f = (dir_f * self.tension_Main_Front)
                self.env_parts['Tension_Main_Front'].set_visibility(True)
                self.env_parts['Tension_Main_Front'].update_arrow(knot_pos_front, knot_pos_front + v_f / 50000.0)
            else: self.env_parts['Tension_Main_Front'].set_visibility(False)
        else: self.env_parts['Tension_Main_Front'].set_visibility(False)
            
        stretch_anc_r = dist_r - self.val_main_line
        if stretch_anc_r > 0: 
            self.tension_Main_Rear = stretch_anc_r * 100000.0
            if self.show_stress_forces:
                dir_r = vec_anc_r_to_center / max(dist_r, 0.001)
                v_r = (dir_r * self.tension_Main_Rear)
                self.env_parts['Tension_Main_Rear'].set_visibility(True)
                self.env_parts['Tension_Main_Rear'].update_arrow(knot_pos_rear, knot_pos_rear + v_r / 50000.0)
            else: self.env_parts['Tension_Main_Rear'].set_visibility(False)
        else: self.env_parts['Tension_Main_Rear'].set_visibility(False)
            
        m_kf = np.eye(4); m_kf[0:3, 3] = knot_pos_front; self.env_parts['Knot_Front'].set_matrix(m_kf)
        m_kr = np.eye(4); m_kr[0:3, 3] = knot_pos_rear; self.env_parts['Knot_Rear'].set_matrix(m_kr)
        self.env_parts['Main_Line_Front'].mesh.points, _ = math_pts_cyl(self.anchor_pos_front, knot_pos_front, 0.1, 0.1)
        self.env_parts['Main_Line_Rear'].mesh.points, _ = math_pts_cyl(self.anchor_pos_rear, knot_pos_rear, 0.1, 0.1)

        target_rpm_L = (self.val_vsurf_L / max(self.Cyl_r, 0.01)) * (60.0 / (2.0 * np.pi))
        target_rpm_R = (self.val_vsurf_R / max(self.Cyl_r, 0.01)) * (60.0 / (2.0 * np.pi))

        torque_net_L = 0.0; torque_net_R = 0.0; frame_thrust_L = 0.0; frame_thrust_R = 0.0
        B_blades = 3.0 
            
        for item in self.dynamic_blades:
            b_id = item['id']; r_id = item['rotor_id']
            is_left = (item['tag'] == 'Left')
            hub_global = hub_L_global if is_left else hub_R_global; rotor_omega = self.omega_L if is_left else self.omega_R
            r_z = rotor_z_L_local if is_left else rotor_z_R_local; r_x = rotor_x_L_local if is_left else rotor_x_R_local
            rot_axis_global = rot_axis_L_global if is_left else rot_axis_R_global
            orbit_a = item['base_angle'] + (self.rotor_angle_L if is_left else self.rotor_angle_R)
            rad_local_flat = r_x * np.cos(orbit_a) + rotor_y_local * np.sin(orbit_a)
            rad_dir_global_flat = R_mat.dot(rad_local_flat)
            
            p_tip_center = hub_global + rad_dir_global_flat * (self.Hub_R + self.Cyl_L)
            item['wire'].mesh.points, _ = math_pts_cyl(hub_global, p_tip_center, 0.03, 0.03)
            
            offset_m = 0.25
            p_cyl_base = hub_global + rad_dir_global_flat * self.Hub_R - rot_axis_global * offset_m
            cyl_vec = p_tip_center - p_cyl_base
            cyl_length = np.linalg.norm(cyl_vec); cyl_vec_dir = cyl_vec / cyl_length
            item['strut'].mesh.points, _ = math_pts_cyl(hub_global, p_cyl_base, 0.1, 0.1)
            
            sweep_dir_cyl = np.cross(rot_axis_global, cyl_vec_dir)
            if np.linalg.norm(sweep_dir_cyl) > 0.001: sweep_dir_cyl /= np.linalg.norm(sweep_dir_cyl)
            normal_dir = np.cross(cyl_vec_dir, sweep_dir_cyl)
            
            m_hardware = np.eye(4)
            m_hardware[0:3, 0] = sweep_dir_cyl; m_hardware[0:3, 1] = normal_dir; m_hardware[0:3, 2] = cyl_vec_dir   
            
            m_bell = np.copy(m_hardware); m_bell[0:3, 3] = p_cyl_base; item['bell_inner'].set_matrix(m_bell)
            m_plate_bot = np.copy(m_hardware); m_plate_bot[0:3, 3] = p_cyl_base; item['plate_bot'].set_matrix(m_plate_bot)
            m_plate_top = np.copy(m_hardware); m_plate_top[0:3, 3] = p_tip_center; item['plate_top'].set_matrix(m_plate_top)
            
            item['blade_cyl'].mesh.points, _ = math_pts_cyl(p_cyl_base, p_tip_center, self.Cyl_r, self.Cyl_r, res=30, rot=0.0)
            item['tape'].mesh.points, _ = math_pts_strip(p_cyl_base, p_tip_center, self.Cyl_r, self.Cyl_r, 0.0, self.cyl_phases[b_id])

            base_rpm = target_rpm_L if is_left else target_rpm_R
            old_cyl_rpm = self.cyl_spins[b_id]
            if self.spinning: self.cyl_spins[b_id] += (base_rpm - self.cyl_spins[b_id]) * dt * 2.0
            else: self.cyl_spins[b_id] *= 0.9
            
            cyl_rpm = self.cyl_spins[b_id]; cyl_omega = cyl_rpm * 0.1047
            alpha_cyl = (cyl_omega - old_cyl_rpm * 0.1047) / dt
            if self.spinning: self.cyl_phases[b_id] += cyl_omega * dt
            
            P_inertia_W = self.I_cyl * alpha_cyl * cyl_omega 
            
            N_strips = 10; dr_strip = cyl_length / N_strips
            total_force = np.array([0.0, 0.0, 0.0]); total_torque_blade = 0.0; total_thrust_comp = 0.0
            mid_strip_idx = N_strips // 2
            
            area_proj_strip = dr_strip * (self.Cyl_r * 2.0)
            AR_geometric = self.Cyl_L / (2.0 * self.Cyl_r)
            
            P_skin_friction_W = 0.0
            vectors = self.blade_vector_parts[b_id]
            
            if not self.show_flow_vectors:
                vectors['flow_A'].set_visibility(False); vectors['flow_B'].set_visibility(False)
            
            for step in range(N_strips):
                p_local = p_cyl_base + cyl_vec_dir * (step + 0.5) * dr_strip
                vec_hub_to_p = p_local - hub_global; r_local_flat = np.dot(vec_hub_to_p, rad_dir_global_flat)
                
                strip_pos_frac = (step + 0.5) / N_strips
                is_ribbed = strip_pos_frac >= (1.0 - self.val_rib_frac)
                
                if is_ribbed: Cf_local = 0.015 
                else: Cf_local = 0.005 
                
                a = item['strips'][step].get('a', 0.0); a_prime = item['strips'][step].get('a_prime', 0.0)
                CL = 0.0; CD_total = 0.6; spin_ratio = 0.01
                
                v_axial_vec = np.array([0, self.val_water_speed * (1 - a), 0])
                v_blade_local = np.cross(rot_axis_global * rotor_omega, rad_dir_global_flat * r_local_flat)
                v_tang_swirl = v_blade_local * a_prime
                app_vec_local = v_axial_vec - v_blade_local - v_tang_swirl - self.sys_vel
                v_app_spanwise = np.dot(app_vec_local, cyl_vec_dir) * cyl_vec_dir
                v_app_perp = app_vec_local - v_app_spanwise
                v_app_perp_mag = np.linalg.norm(v_app_perp); safe_v_app_perp = max(v_app_perp_mag, 0.01)
                v_app_perp_dir = v_app_perp / safe_v_app_perp
                
                flow_downstream_dir = v_app_perp_dir 
                
                current_flap_dir = item['strips'][step]['flap_dir']
                if current_flap_dir is None: current_flap_dir = flow_downstream_dir
                
                response_rate = 2.0 
                damped_dir = current_flap_dir + (flow_downstream_dir - current_flap_dir) * (response_rate * dt)
                damped_dir_norm = np.linalg.norm(damped_dir)
                if damped_dir_norm > 0.001: damped_dir /= damped_dir_norm
                else: damped_dir = flow_downstream_dir
                item['strips'][step]['flap_dir'] = damped_dir
                
                v_x = np.dot(damped_dir, normal_dir)
                v_y = np.dot(damped_dir, sweep_dir_cyl)
                theta_flap = np.arctan2(v_y, v_x)
                
                if not self.freeze_base:
                    item['frozen_twist'][step] = theta_flap
                    theta_base = theta_flap
                    if step == mid_strip_idx: item['current_base_ref'] = 0.0
                else:
                    if step == mid_strip_idx: item['current_base_ref'] = theta_flap - item['frozen_twist'][step]
                    theta_base = item['current_base_ref'] + item['frozen_twist'][step]
                
                vec_base = normal_dir * np.cos(theta_base) + sweep_dir_cyl * np.sin(theta_base)
                vec_flap = normal_dir * np.cos(theta_flap) + sweep_dir_cyl * np.sin(theta_flap)
                
                diff = (theta_flap - theta_base + np.pi) % (2 * np.pi) - np.pi
                mismatch_rad = abs(diff)
                penalty = np.clip(mismatch_rad / 0.5, 0.0, 1.0) 
                total_aero_penalty += penalty
                
                if self.spinning:
                    for _ in range(15):
                        v_ax = max(self.val_water_speed * (1 - a), 0.01)
                        v_tg = abs(rotor_omega * r_local_flat * (1 + a_prime))
                        phi = np.arctan2(v_ax, max(v_tg, 0.01))
                        v_app_bem = np.sqrt(v_ax**2 + v_tg**2)
                        
                        f_tip = (B_blades / 2.0) * (self.Hub_R + self.Cyl_L - r_local_flat) / (r_local_flat * max(np.sin(phi), 0.01))
                        F_prandtl = (2.0 / np.pi) * np.arccos(np.clip(np.exp(-f_tip), -1.0, 1.0))
                        F_prandtl = np.clip(np.nan_to_num(F_prandtl), 0.001, 1.0)
                        
                        spin_ratio = abs(cyl_omega * self.Cyl_r) / max(v_app_bem, 0.01)
                        
                        if self.show_flaps:
                            mean_CL = np.interp(spin_ratio, self.lut_sr_flap, self.lut_cl_flap)
                            mean_CD = np.interp(spin_ratio, self.lut_sr_flap, self.lut_cd_flap)
                            mean_CL *= max(0.2, 1.0 - 0.40 * penalty) 
                            mean_CD *= (1.0 + 0.20 * penalty)
                        else:
                            mean_CL = np.interp(spin_ratio, self.lut_sr_base, self.lut_cl_base)
                            mean_CD = np.interp(spin_ratio, self.lut_sr_base, self.lut_cd_base)

                        if is_ribbed:
                            mean_CL *= (self.val_rib_cl_max / 5.0) 
                            mean_CD *= 1.30
                            
                        St = 0.20 
                        D = 2.0 * self.Cyl_r
                        f_vortex = (St * safe_v_app_perp) / D 
                        
                        wake_damping = max(0.0, 1.0 - (spin_ratio / 2.0))
                        if self.show_flaps: wake_damping *= 0.2
                            
                        amp_CL = 0.15 * mean_CL * wake_damping
                        amp_CD = 0.10 * mean_CD * wake_damping
                        
                        CL = mean_CL + amp_CL * np.sin(2 * np.pi * f_vortex * self.sim_time)
                        CD_profile = mean_CD + amp_CD * np.sin(2 * np.pi * (2.0 * f_vortex) * self.sim_time)
                        
                        e_oswald = 0.45
                        R_plate = self.Cyl_r * 2.5
                        plate_effect = 1.0 + 1.9 * ((R_plate - self.Cyl_r) / self.Cyl_r)
                        AR_effective = AR_geometric * plate_effect
                        
                        CD_induced = (CL**2) / (np.pi * AR_effective * e_oswald)
                        CD_total = CD_profile + CD_induced
                            
                        sigma = (B_blades * 2 * self.Cyl_r) / (2 * np.pi * r_local_flat)
                        
                        mean_CD_total = mean_CD + CD_induced 
                        
                        Cx_mean = mean_CL * np.cos(phi) + mean_CD_total * np.sin(phi)
                        Cy_mean = mean_CL * np.sin(phi) - mean_CD_total * np.cos(phi)
                        
                        CT_local = (sigma * Cx_mean * (1 - a)**2) / max(np.sin(phi)**2, 1e-4)
                        CT_modified = CT_local / max(F_prandtl, 0.001)
                        
                        a_new = np.interp(CT_modified, self.lut_ct, self.lut_a)
                        a_prime_new = 1.0 / ((4 * F_prandtl * np.sin(phi) * np.cos(phi)) / (sigma * max(Cy_mean, 1e-4)) - 1)
                        
                        a = 0.5 * a + 0.5 * np.clip(np.nan_to_num(a_new), 0.0, 0.85)
                        a_prime = 0.5 * a_prime + 0.5 * np.clip(np.nan_to_num(a_prime_new), -0.5, 0.5)

                item['strips'][step]['a'] = a; item['strips'][step]['a_prime'] = a_prime
                if b_id == 0 and step == mid_strip_idx: self.SR_L = spin_ratio; self.CL_L = CL; self.CD_L = CD_total; self.a_L = a
                elif b_id == 3 and step == mid_strip_idx: self.SR_R = spin_ratio; self.CL_R = CL; self.CD_R = CD_total; self.a_R = a

                gap = 0.12 
                flap_width = 0.25 
                half_w = dr_strip * 0.45 
                
                p_hinge = p_local + vec_base * (self.Cyl_r + gap)
                
                item['rod_pts'][step * 2] = p_hinge - cyl_vec_dir * half_w * 1.1
                item['rod_pts'][step * 2 + 1] = p_hinge + cyl_vec_dir * half_w * 1.1
                
                if self.show_flaps:
                    p1 = p_hinge - cyl_vec_dir * half_w
                    p2 = p_hinge + cyl_vec_dir * half_w
                    p3 = p1 + vec_flap * flap_width
                    p4 = p2 + vec_flap * flap_width
                    
                    item['flaps'][step].mesh.points = np.array([p1, p2, p3, p4])
                    if self.spinning: item['flaps'][step].set_visibility(True)
                else:
                    item['flaps'][step].set_visibility(False)

                v_surface_out = abs(cyl_omega * self.Cyl_r)
                area_strip_out = 2 * np.pi * self.Cyl_r * dr_strip
                P_skin_friction_W += self.rho * Cf_local * area_strip_out * (v_surface_out**3)

                k1 = 0.020 if is_ribbed else 0.012; k2 = 0.005 if is_ribbed else 0.002
                C_M = k1 * spin_ratio + k2 * (spin_ratio**2)
                dQ_pressure = C_M * (0.5 * self.rho * (safe_v_app_perp**2) * area_proj_strip) * self.Cyl_r
                P_skin_friction_W += dQ_pressure * abs(cyl_omega)

                if step == mid_strip_idx and self.show_flow_vectors and self.val_water_speed > 0.1:
                    fluid_dir = v_app_perp_dir; base_mag = safe_v_app_perp
                    side_dir = np.cross(cyl_vec_dir, fluid_dir)
                    if np.linalg.norm(side_dir) > 0.001: side_dir /= np.linalg.norm(side_dir)
                    p_A = p_local + side_dir * (self.Cyl_r * 1.5); p_B = p_local - side_dir * (self.Cyl_r * 1.5)
                    omega_vec = cyl_vec_dir * cyl_omega; v_surf_A = np.cross(omega_vec, side_dir * self.Cyl_r)
                    mod_A = np.dot(v_surf_A, fluid_dir); decay = 0.66 
                    mag_A = base_mag + mod_A * decay; mag_B = base_mag - mod_A * decay
                    vec_A = fluid_dir * max(0.1, mag_A); vec_B = fluid_dir * max(0.1, mag_B)
                    vectors['flow_A'].set_visibility(True); vectors['flow_A'].update_arrow(p_A - vec_A*0.5, p_A + vec_A*0.5)
                    vectors['flow_B'].set_visibility(True); vectors['flow_B'].update_arrow(p_B - vec_B*0.5, p_B + vec_B*0.5)

                spin_axis_local = cyl_vec_dir * np.sign(cyl_rpm) if cyl_rpm != 0 else cyl_vec_dir
                mag_lift_dir = np.cross(v_app_perp_dir, spin_axis_local)
                if np.linalg.norm(mag_lift_dir) > 0.001: mag_lift_dir /= np.linalg.norm(mag_lift_dir)
                
                dF_magnus = mag_lift_dir * (0.5 * self.rho * (safe_v_app_perp**2) * area_proj_strip * CL)
                dF_drag = v_app_perp_dir * (0.5 * self.rho * (safe_v_app_perp**2) * area_proj_strip * CD_total)
                dF_total = dF_magnus + dF_drag
                
                total_force += dF_total; total_torque_blade += np.dot(dF_total, sweep_dir_cyl) * r_local_flat
                total_thrust_comp += np.dot(dF_total, rot_axis_global); Sys_Net_Torque += np.cross(p_local - self.sys_pos, dF_total)

            item['rod'].mesh.points = item['rod_pts']
            if self.show_flaps and self.spinning: item['rod'].set_visibility(True)
            else: item['rod'].set_visibility(False)

            Sys_Net_Force += total_force
            force_drive_vec = sweep_dir_cyl * np.dot(total_force, sweep_dir_cyl)
            force_axial_vec = rot_axis_global * np.dot(total_force, rot_axis_global)
            
            p_mid_cyl = p_cyl_base + cyl_vec_dir * (self.Cyl_L / 2.0)
            if self.spinning and np.linalg.norm(total_force) > 50.0:
                force_scale = 4000.0 
                if self.show_force_total:
                    vectors['force_total'].set_visibility(True); vectors['force_total'].update_arrow(p_mid_cyl, p_mid_cyl + total_force / force_scale)
                else: vectors['force_total'].set_visibility(False)
                
                if self.show_force_components:
                    vectors['force_drive'].set_visibility(True); vectors['force_axial'].set_visibility(True)
                    vectors['force_drive'].update_arrow(p_mid_cyl, p_mid_cyl + force_drive_vec / force_scale)
                    vectors['force_axial'].update_arrow(p_mid_cyl, p_mid_cyl + force_axial_vec / force_scale)
                else:
                    vectors['force_drive'].set_visibility(False); vectors['force_axial'].set_visibility(False)
            else:
                vectors['force_total'].set_visibility(False); vectors['force_drive'].set_visibility(False); vectors['force_axial'].set_visibility(False)
            
            F_centrifugal_mag = self.Cyl_Mass * (rotor_omega**2) * self.Cyl_Mid_R
            F_centrifugal_vec = rad_dir_global_flat * F_centrifugal_mag
            if self.show_stress_forces and self.spinning:
                stress_scale = 20000.0 
                vectors['stress_tip'].set_visibility(True); vectors['stress_tip'].update_arrow(p_tip_center, p_tip_center + F_centrifugal_vec / stress_scale)
                bending_force = force_axial_vec + force_drive_vec
                vectors['stress_hub'].set_visibility(True); vectors['stress_hub'].update_arrow(p_cyl_base, p_cyl_base + bending_force / stress_scale)
            else:
                vectors['stress_tip'].set_visibility(False); vectors['stress_hub'].set_visibility(False)

            if is_left: torque_net_L += total_torque_blade; frame_thrust_L += total_thrust_comp
            else: torque_net_R += total_torque_blade; frame_thrust_R += total_thrust_comp
                
            v_surface_in = abs(cyl_omega * self.r_in)
            P_skin_friction_W += self.rho * 0.005 * (2 * np.pi * self.r_in * self.Cyl_L) * (v_surface_in**3)
            P_mech_W = P_skin_friction_W + P_inertia_W 
            
            if self.spinning and P_mech_W > 0:
                rated_motor_power_W = 15000.0 
                current_load_pct = min(P_mech_W / rated_motor_power_W, 1.2)
                
                actual_mot_eff = np.interp(current_load_pct, self.lut_load_pct, self.lut_eff_mot)
                actual_mot_eff = max(actual_mot_eff, 0.05)
                
                if is_left: 
                    self.P_mot_L += P_mech_W / actual_mot_eff
                    self.eff_mot_L = actual_mot_eff
                else: 
                    self.P_mot_R += P_mech_W / actual_mot_eff
                    self.eff_mot_R = actual_mot_eff

        self.thrust_L = frame_thrust_L; self.thrust_R = frame_thrust_R
        self.drive_L = torque_net_L / self.Cyl_Mid_R if self.Cyl_Mid_R > 0 else 0; self.drive_R = torque_net_R / self.Cyl_Mid_R if self.Cyl_Mid_R > 0 else 0
        self.avg_flap_penalty = total_aero_penalty / 60.0 
        
        Area_surge = 12.0; Area_heave_pitch = 100.0; CD_frame = 1.1 
        F_hydro_drag = -0.5 * self.rho * CD_frame * Area_surge * self.sys_vel * np.abs(self.sys_vel)
        lever_arm = self.Frame_Depth / 2.0; v_pitch_ends = self.sys_omega_pitch * lever_arm
        T_hydro_damp_pitch = -2.0 * (0.5 * self.rho * CD_frame * Area_heave_pitch * v_pitch_ends * np.abs(v_pitch_ends)) * lever_arm
        v_yaw_ends = self.sys_omega_yaw * (self.Hub_Dist_Mean / 2.0)
        T_hydro_damp_yaw = -2.0 * (0.5 * self.rho * CD_frame * Area_surge * v_yaw_ends * np.abs(v_yaw_ends)) * (self.Hub_Dist_Mean / 2.0)

        accel = (Sys_Net_Force + F_hydro_drag) / self.System_Mass
        accel = np.clip(np.nan_to_num(accel), -50.0, 50.0)
        self.sys_vel += accel * dt
        self.sys_pos += self.sys_vel * dt
        self.sys_pos[2] = min(self.sys_pos[2], -1.0) 

        alpha_pitch = (Sys_Net_Torque[0] + T_hydro_damp_pitch) / self.System_Inertia_Pitch
        alpha_pitch = np.clip(np.nan_to_num(alpha_pitch), -10.0, 10.0)
        self.sys_omega_pitch += alpha_pitch * dt
        self.sys_pitch += self.sys_omega_pitch * dt

        alpha_yaw = (Sys_Net_Torque[2] + T_hydro_damp_yaw) / self.System_Inertia_Yaw
        alpha_yaw = np.clip(np.nan_to_num(alpha_yaw), -10.0, 10.0)
        self.sys_omega_yaw += alpha_yaw * dt
        self.sys_yaw += self.sys_omega_yaw * dt
        
        self.sys_vel *= 0.95
        self.sys_omega_pitch *= 0.90
        self.sys_omega_yaw *= 0.90

        if self.spinning:
            gen_max_torque = 2500000.0 
            app_torque = gen_max_torque * (self.val_gen_load / 100.0)
            frame_drag_L = 40000.0 * self.omega_L * abs(self.omega_L); frame_drag_R = 40000.0 * self.omega_R * abs(self.omega_R)
            brake_L = app_torque * np.sign(self.omega_L) if abs(self.omega_L) > 0.05 else 0.0
            brake_R = app_torque * np.sign(self.omega_R) if abs(self.omega_R) > 0.05 else 0.0
            
            self.P_gen_mech_L = abs(brake_L * self.omega_L); self.P_gen_mech_R = abs(brake_R * self.omega_R)
            
            rated_gen_power_W = 150000.0 
            load_pct_L = min(self.P_gen_mech_L / rated_gen_power_W, 1.2) if rated_gen_power_W > 0 else 0
            load_pct_R = min(self.P_gen_mech_R / rated_gen_power_W, 1.2) if rated_gen_power_W > 0 else 0
            
            self.eff_gen_L = np.interp(load_pct_L, self.lut_load_pct, self.lut_eff_gen)
            self.eff_gen_R = np.interp(load_pct_R, self.lut_load_pct, self.lut_eff_gen)
            
            self.P_gen_elec_L = self.P_gen_mech_L * self.eff_gen_L
            self.P_gen_elec_R = self.P_gen_mech_R * self.eff_gen_R
            
            net_L = torque_net_L - frame_drag_L - brake_L; net_R = torque_net_R - frame_drag_R - brake_R
            alpha_L = net_L / self.Inertia_Rotor; alpha_R = net_R / self.Inertia_Rotor
            
            alpha_L = np.clip(np.nan_to_num(alpha_L), -100.0, 100.0)
            alpha_R = np.clip(np.nan_to_num(alpha_R), -100.0, 100.0)
            
            self.omega_L += alpha_L * dt; self.omega_R += alpha_R * dt
            if abs(self.omega_L) < 0.02 and abs(net_L) < app_torque + 100: self.omega_L = 0.0
            if abs(self.omega_R) < 0.02 and abs(net_R) < app_torque + 100: self.omega_R = 0.0
            self.rotor_angle_L += self.omega_L * dt; self.rotor_angle_R += self.omega_R * dt
        else:
            self.omega_L *= 0.95; self.omega_R *= 0.95
            self.rotor_angle_L += self.omega_L * dt; self.rotor_angle_R += self.omega_R * dt

        self.mech_P_gen_W = self.P_gen_mech_L + self.P_gen_mech_R
        self.total_P_gen_W = self.P_gen_elec_L + self.P_gen_elec_R
        self.total_P_mot_W = self.P_mot_L + self.P_mot_R
        self.net_P_W = self.total_P_gen_W - self.total_P_mot_W

        if self.autopilot_on and self.spinning:
            pitch_deg = np.degrees(self.sys_pitch)
            if abs(pitch_deg) > 0.5:
                self.val_winch_pitch -= pitch_deg * dt * 0.5
                self.val_winch_pitch = np.clip(self.val_winch_pitch, -15.0, 15.0)
                self.slider_pitch.GetRepresentation().SetValue(self.val_winch_pitch)

            if abs(self.val_water_speed - self.ap_prev_water) > 0.1:
                new_vsurf = np.interp(self.val_water_speed, self.lut_v, self.lut_vsurf)
                new_load = np.interp(self.val_water_speed, self.lut_v, self.lut_load)
                
                self.val_vsurf_L = float(new_vsurf)
                self.val_vsurf_R = -self.val_vsurf_L
                self.val_gen_load = float(new_load)
                
                self.slider_spin_L.GetRepresentation().SetValue(self.val_vsurf_L)
                self.slider_spin_R.GetRepresentation().SetValue(self.val_vsurf_R)
                self.slider_load.GetRepresentation().SetValue(self.val_gen_load)
                
                self.ap_prev_water = self.val_water_speed
                self.ap_timer = 0.0 

            self.ap_timer += dt
            if self.ap_timer >= self.ap_interval:
                current_net = self.net_P_W
                delta_P = current_net - self.ap_prev_net
                
                if abs(delta_P) > 5.0: 
                    if self.ap_phase == 'SPIN':
                        if delta_P < 0: self.ap_dir_load *= -1
                    else:
                        if delta_P < 0: self.ap_dir_spin *= -1
                
                if self.ap_phase == 'SPIN':
                    self.val_vsurf_L += self.ap_dir_spin * self.ap_step_vsurf
                    self.val_vsurf_L = np.clip(self.val_vsurf_L, 1.0, 15.0)
                    self.val_vsurf_R = -self.val_vsurf_L
                    self.slider_spin_L.GetRepresentation().SetValue(self.val_vsurf_L)
                    self.slider_spin_R.GetRepresentation().SetValue(self.val_vsurf_R)
                    self.ap_phase = 'LOAD'
                else:
                    self.val_gen_load += self.ap_dir_load * self.ap_step_load
                    self.val_gen_load = np.clip(self.val_gen_load, 5, 100)
                    self.slider_load.GetRepresentation().SetValue(self.val_gen_load)
                    self.ap_phase = 'SPIN'
                    
                self.ap_prev_net = self.net_P_W
                self.ap_timer = 0.0

        self.update_hud()

    def export_cfd_gui(self, state):
        if not state: return  
        
        water_v = max(self.val_water_speed, 0.1)

        try:
            print("\n[CFD] Άνοιγμα παραθύρου ρυθμίσεων...")
            
            # --- UNIVERSAL ROUTING ---
            if getattr(sys, 'frozen', False):
                # Αν τρέχει ως .exe
                cmd = [sys.executable, '--run-cfd-dialog', str(water_v)]
            else:
                # Αν τρέχει ως .py 
                cmd = [sys.executable, os.path.abspath(__file__), '--run-cfd-dialog', str(water_v)]
                
            result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as e:
            print("[CFD] Σφάλμα ανοίγματος διαλόγου:", e)
            return
        
        if result == 'CANCEL' or '|' not in result: 
            print("[CFD] Η εξαγωγή ακυρώθηκε από τον χρήστη.")
            return
        
        try:
            dur_str, fps_str = result.split('|')
            duration = float(dur_str)
            fps = float(fps_str)
            write_interval = round(1.0 / fps, 4)
        except Exception as e: 
            print("[CFD] Λάθος ανάγνωσης δεδομένων:", e)
            return

        print("\n" + "="*60)
        print(f" [CFD EXPORTER] FULL TRANSIENT 6-DOF KINEMATICS")
        print(f" -> Χρόνος: {duration}s | 3D Output κάθε: {write_interval}s ({fps} FPS)")
        print("="*60)

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"CFD_Export_6DoF_{self.val_water_speed}ms_{timestamp}"
            os.makedirs(folder_name, exist_ok=True)
            print(f"[1/6] Δημιουργήθηκε ο φάκελος: {folder_name}")

            # --- ΕΞΑΓΩΓΗ FULL PHYSICS STATE ---
            physics_state = {
                "inputs": {
                    "val_water_speed": self.val_water_speed, "val_main_line": self.val_main_line,
                    "val_winch_pitch": self.val_winch_pitch, "val_winch_yaw": self.val_winch_yaw,
                    "val_vsurf_L": self.val_vsurf_L, "val_vsurf_R": self.val_vsurf_R,
                    "val_gen_load": self.val_gen_load, "val_cyl_L": self.val_cyl_L,
                    "val_cyl_r": self.val_cyl_r, "val_rib_frac": self.val_rib_frac,
                    "val_rib_cl_max": self.val_rib_cl_max, "val_rib_sr_peak": self.val_rib_sr_peak
                },
                "kinematics": {
                    "sys_pos": self.sys_pos.tolist(), "sys_vel": self.sys_vel.tolist(),
                    "sys_pitch": float(self.sys_pitch), "sys_omega_pitch": float(self.sys_omega_pitch),
                    "sys_yaw": float(self.sys_yaw), "sys_omega_yaw": float(self.sys_omega_yaw)
                },
                "rotors": {
                    "omega_L": self.omega_L, "omega_R": self.omega_R,
                    "rotor_angle_L": self.rotor_angle_L, "rotor_angle_R": self.rotor_angle_R,
                    "cyl_spins": self.cyl_spins, "cyl_phases": self.cyl_phases
                },
                "time": self.sim_time
            }
            with open(os.path.join(folder_name, "scenario_state.json"), "w", encoding="utf-8") as f:
                json.dump(physics_state, f, indent=4)

            def get_poly(part):
                poly = part.mesh.extract_surface(algorithm=None).copy()
                poly.transform(part.actor.user_matrix, inplace=True)
                return poly

            print("[2/6] Εξαγωγή Ανεξάρτητων Γεωμετριών για το Overset...")
            os.makedirs(os.path.join(folder_name, "constant", "triSurface"), exist_ok=True)
            
            # 1. STATOR (Πλωτήρες)
            stator_blocks = pv.MultiBlock()
            for key, part in self.frame_parts.items():
                if part.user_visible and part.mesh.n_points > 0: stator_blocks.append(get_poly(part))
            if len(stator_blocks) > 0:
                s_m = stator_blocks.combine()
                if not isinstance(s_m, pv.PolyData): s_m = s_m.extract_surface(algorithm=None)
                s_m.save(os.path.join(folder_name, "constant", "triSurface", "stator.stl"))

            cp = np.cos(self.sys_pitch); sp = np.sin(self.sys_pitch)
            cy = np.cos(self.sys_yaw);   sy = np.sin(self.sys_yaw)
            Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]); Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
            R_mat = Rz.dot(Rx)
            axis_L = R_mat.dot(np.array([np.sin(self.toe_in_angle), -np.cos(self.toe_in_angle), 0]))
            axis_R = R_mat.dot(np.array([-np.sin(self.toe_in_angle), -np.cos(self.toe_in_angle), 0]))
            
            x_hub = (self.Hub_Dist_Mean/2)
            hub_L_global = self.sys_pos + R_mat.dot(np.array([-x_hub, 0.0, 0]))
            hub_R_global = self.sys_pos + R_mat.dot(np.array([ x_hub, 0.0, 0]))

            cyl_kinematics = []

            # 2. HUBS & 3. CYLINDERS
            for r_tag, is_left in [('left', True), ('right', False)]:
                hub_blocks = pv.MultiBlock()
                hub_part = self.rotor_parts[0 if is_left else 1]
                for key, part in hub_part.items():
                    if part.user_visible and part.mesh.n_points > 0: hub_blocks.append(get_poly(part))
                
                cyl_idx = 0
                for item in self.dynamic_blades:
                    if (item['tag'] == 'Left') == is_left:
                        for pk in ['strut', 'bell_inner', 'wire', 'rod']:
                            if pk in item and item[pk].user_visible: hub_blocks.append(get_poly(item[pk]))
                        if self.show_flaps:
                            for flap in item['flaps']:
                                if flap.user_visible: hub_blocks.append(get_poly(flap))
                        
                        if item['blade_cyl'].user_visible:
                            c_blocks = pv.MultiBlock()
                            c_blocks.append(get_poly(item['blade_cyl']))
                            c_blocks.append(get_poly(item['plate_top']))
                            c_blocks.append(get_poly(item['plate_bot']))
                            c_m = c_blocks.combine()
                            if not isinstance(c_m, pv.PolyData): c_m = c_m.extract_surface(algorithm=None)
                            c_name = f"cyl_{r_tag}_{cyl_idx}"
                            c_m.save(os.path.join(folder_name, "constant", "triSurface", f"{c_name}.stl"))
                            
                            bounds = c_m.bounds
                            c_center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, (bounds[4]+bounds[5])/2]
                            cyl_vec_dir = item['blade_cyl'].actor.user_matrix[0:3, 2] 
                            cyl_kinematics.append({
                                'name': c_name, 'C_hub': hub_L_global if is_left else hub_R_global,
                                'A_hub': axis_L if is_left else axis_R, 'W_orb': self.omega_L if is_left else self.omega_R,
                                'C0_cyl': c_center, 'A_cyl': cyl_vec_dir, 'W_spin': self.cyl_spins[item['id']] * 0.1047
                            })
                            cyl_idx += 1

                if len(hub_blocks) > 0:
                    h_m = hub_blocks.combine()
                    if not isinstance(h_m, pv.PolyData): h_m = h_m.extract_surface(algorithm=None)
                    h_m.save(os.path.join(folder_name, "constant", "triSurface", f"hub_{r_tag}.stl"))

            print("[3/6] Μαθηματικός Υπολογισμός 6-DoF Kinematics...")
            def write_6dof(filepath, C_hub, A_hub, W_orb, C0_cyl, A_cyl, W_spin, dur):
                def get_R(k, theta):
                    k = np.array(k, dtype=float); k /= np.linalg.norm(k)
                    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                    return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*K.dot(K)
                
                dt_step = 0.02 
                t_arr = np.arange(0, dur + dt_step*2, dt_step)
                with open(filepath, "w") as f:
                    f.write("(\n")
                    for t in t_arr:
                        R_orb = get_R(A_hub, W_orb * t)
                        R_spin = get_R(A_cyl, W_spin * t)
                        R_tot = R_orb.dot(R_spin)
                        
                        V0 = np.array(C0_cyl) - np.array(C_hub)
                        C_t = np.array(C_hub) + R_orb.dot(V0)
                        
                        Rf = R_tot.flatten()
                        f.write(f"    ({t:.4f} (({C_t[0]:.6f} {C_t[1]:.6f} {C_t[2]:.6f}) ({Rf[0]:.6f} {Rf[1]:.6f} {Rf[2]:.6f} {Rf[3]:.6f} {Rf[4]:.6f} {Rf[5]:.6f} {Rf[6]:.6f} {Rf[7]:.6f} {Rf[8]:.6f})))\n")
                    f.write(")\n")

            for c in cyl_kinematics:
                write_6dof(os.path.join(folder_name, "constant", f"motion_{c['name']}.dat"), 
                           c['C_hub'], c['A_hub'], c['W_orb'], c['C0_cyl'], c['A_cyl'], c['W_spin'], duration)

            print("[4/6] Δημιουργία OpenFOAM Dictionaries...")
            os.makedirs(os.path.join(folder_name, "0.orig"), exist_ok=True)
            os.makedirs(os.path.join(folder_name, "system"), exist_ok=True)

            X_span = self.Hub_Dist_Mean + (self.Cyl_L * 2)
            X_min, X_max = -(X_span/2) - 15.0, (X_span/2) + 15.0
            Y_min, Y_max = -15.0, 50.0
            Z_min, Z_max = self.sys_pos[2] - (self.Hub_R+self.Cyl_L) - 10.0, 10.0
            Nx, Ny, Nz = int((X_max-X_min)/1.5), int((Y_max-Y_min)/1.5), int((Z_max-Z_min)/1.5)

            bm_dict = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class dictionary; object blockMeshDict; }}
\\*---------------------------------------------------------------------------*/
scale   1;
vertices (
    ({X_min} {Y_min} {Z_min}) ({X_max} {Y_min} {Z_min}) ({X_max} {Y_max} {Z_min}) ({X_min} {Y_max} {Z_min})
    ({X_min} {Y_min} {Z_max}) ({X_max} {Y_min} {Z_max}) ({X_max} {Y_max} {Z_max}) ({X_min} {Y_max} {Z_max})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({Nx} {Ny} {Nz}) simpleGrading (1 1 1) );
edges ();
boundary (
    inlet {{ type patch; faces ( (0 4 5 1) ); }}
    outlet {{ type patch; faces ( (3 2 6 7) ); }}
    sides {{ type symmetry; faces ( (1 5 6 2) (0 3 7 4) (0 1 2 3) (4 7 6 5) ); }}
);
mergePatchPairs ();
"""
            with open(os.path.join(folder_name, "system", "blockMeshDict"), "w") as f: f.write(bm_dict)

            control_dict_content = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class dictionary; object controlDict; }}
\\*---------------------------------------------------------------------------*/
application     overPimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {duration};
deltaT          0.001;
writeControl    runTime;
writeInterval   {write_interval}; // <-- Προέρχεται από το Slider του FPS!
maxCo           1.5;

functions
{{
    forces_left
    {{
        type            forceCoeffs;
        libs            ("libforces.so");
        writeControl    timeStep;
        writeInterval   1;
        patches         ("cyl_left_0" "cyl_left_1" "cyl_left_2" "hub_left");
        rho             rhoInf;
        rhoInf          {self.rho};
        liftDir         (1 0 0);
        dragDir         (0 1 0);
        pitchAxis       (0 0 1);
        magUInf         {self.val_water_speed:.3f};
        lRef            {self.Cyl_L};
        Aref            {self.Cyl_L * (2*self.Cyl_r) * 3};
    }}
}}
"""
            with open(os.path.join(folder_name, "system", "controlDict"), "w") as f: f.write(control_dict_content)

            dyn_mesh = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class dictionary; object dynamicMeshDict; }}
\\*---------------------------------------------------------------------------*/
dynamicFvMesh   dynamicOversetFvMesh;
motionSolverLibs ( "libfvMotionSolvers.so" "libsixDoFRigidBodyMotion.so" );
motionSolver    solidBody;

solidBodyMotionFunctions
{{
    hub_left {{
        solidBodyMotionFunction rotatingMotion;
        origin ({hub_L_global[0]:.5f} {hub_L_global[1]:.5f} {hub_L_global[2]:.5f});
        axis   ({axis_L[0]:.5f} {axis_L[1]:.5f} {axis_L[2]:.5f});
        omega  {self.omega_L:.5f};
    }}
    hub_right {{
        solidBodyMotionFunction rotatingMotion;
        origin ({hub_R_global[0]:.5f} {hub_R_global[1]:.5f} {hub_R_global[2]:.5f});
        axis   ({axis_R[0]:.5f} {axis_R[1]:.5f} {axis_R[2]:.5f});
        omega  {self.omega_R:.5f};
    }}
"""
            for c in cyl_kinematics:
                dyn_mesh += f"""    {c['name']} {{ solidBodyMotionFunction tabulated6DoFMotion; timeDataFileName "constant/motion_{c['name']}.dat"; }}\n"""
            dyn_mesh += "}\n"
            with open(os.path.join(folder_name, "constant", "dynamicMeshDict"), "w") as f: f.write(dyn_mesh)

            u_str = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class volVectorField; object U; }}
\\*---------------------------------------------------------------------------*/
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 {self.val_water_speed:.3f} 0);
boundaryField
{{
    inlet   {{ type fixedValue; value uniform (0 {self.val_water_speed:.3f} 0); }}
    outlet  {{ type zeroGradient; }}
    sides   {{ type symmetry; }}
    stator  {{ type noSlip; }}
    hub_left  {{ type movingWallVelocity; value uniform (0 0 0); }}
    hub_right {{ type movingWallVelocity; value uniform (0 0 0); }}
"""
            for c in cyl_kinematics:
                u_str += f"    {c['name']} {{ type movingWallVelocity; value uniform (0 0 0); }}\n"
            u_str += "}\n"
            with open(os.path.join(folder_name, "0.orig", "U"), "w") as f: f.write(u_str)

            print("[5/6] Δημιουργία cellZones (topoSetDict)...")
            r_overset = self.Hub_R + self.Cyl_L + 2.0
            topo_str = f"""/*--------------------------------*- C++ -*----------------------------------*\\
FoamFile {{ version 2.0; format ascii; class dictionary; object topoSetDict; }}
\\*---------------------------------------------------------------------------*/
actions (
    {{ name z_hubL; type cellSet; action new; source cylinderToCell; sourceInfo {{ p1 ({hub_L_global[0]-1} {hub_L_global[1]} {hub_L_global[2]}); p2 ({hub_L_global[0]+1} {hub_L_global[1]} {hub_L_global[2]}); radius {r_overset}; }} }}
    {{ name hub_left; type cellZoneSet; action new; source setToCellZone; sourceInfo {{ set z_hubL; }} }}
    
    {{ name z_hubR; type cellSet; action new; source cylinderToCell; sourceInfo {{ p1 ({hub_R_global[0]-1} {hub_R_global[1]} {hub_R_global[2]}); p2 ({hub_R_global[0]+1} {hub_R_global[1]} {hub_R_global[2]}); radius {r_overset}; }} }}
    {{ name hub_right; type cellZoneSet; action new; source setToCellZone; sourceInfo {{ set z_hubR; }} }}
"""
            for c in cyl_kinematics:
                topo_str += f"""    {{ name z_{c['name']}; type cellSet; action new; source cylinderToCell; sourceInfo {{ p1 ({c['C0_cyl'][0]-0.5} {c['C0_cyl'][1]} {c['C0_cyl'][2]}); p2 ({c['C0_cyl'][0]+0.5} {c['C0_cyl'][1]} {c['C0_cyl'][2]}); radius {self.Cyl_r * 2.5}; }} }}\n    {{ name {c['name']}; type cellZoneSet; action new; source setToCellZone; sourceInfo {{ set z_{c['name']}; }} }}\n"""
            topo_str += ");\n"
            with open(os.path.join(folder_name, "system", "topoSetDict"), "w") as f: f.write(topo_str)

            print("[6/6] Δημιουργία Smart Bash Script (Auto-Resume)...")
            bash_script = f"""#!/bin/bash
echo "=== River-Monster: FULL TRANSIENT 6-DOF overPimpleFoam ==="

MAX_CORES=${{NUMBER_OF_PROCESSORS:-$(nproc)}}
read -p "Πυρήνες (Max $MAX_CORES): " USER_CORES
NCORES=${{USER_CORES:-$MAX_CORES}}

mkdir -p system constant/triSurface 0.orig

# ΛΟΓΙΚΗ AUTO-RESUME
if [ -d "processor0" ]; then
    echo "======================================================"
    echo " [AUTO-RESUME] Εντοπίστηκε προηγούμενο run!"
    echo " Η προσομοίωση θα συνεχιστεί από το τελευταίο time step."
    echo "======================================================"
    
    # Αλλάζει δυναμικά το controlDict για να ξεκινήσει από το latestTime
    sed -i 's/startFrom[ \t]*startTime;/startFrom        latestTime;/g' system/controlDict
    
    echo "-> Συνέχιση Επίλυσης (overPimpleFoam σε $NCORES πυρήνες)..."
    # Το >> κάνει append (προσθήκη) στο υπάρχον run.log αντί να το διαγράψει
    mpiexec -n $NCORES overPimpleFoam -parallel >> run.log 2>&1

else
    echo "======================================================"
    echo " [FRESH START] Καθαρή εκκίνηση νέας προσομοίωσης."
    echo "======================================================"
    
    cp -n $FOAM_TUTORIALS/incompressible/overPimpleFoam/twoSimpleRotors/system/* ./system/ 2>/dev/null
    cp -n $FOAM_TUTORIALS/incompressible/overPimpleFoam/twoSimpleRotors/constant/* ./constant/ 2>/dev/null

    cat <<EOF > system/decomposeParDict
FoamFile {{ version 2.0; format ascii; class dictionary; object decomposeParDict; }}
numberOfSubdomains  $NCORES;
method              scotch;
EOF

    echo "-> 1. Μεταφορά STLs..."
    mv *.stl constant/triSurface/ 2>/dev/null

    echo "-> 2. Meshing (blockMesh & snappyHexMesh)..."
    surfaceFeatureExtract > log.surfaceFeatureExtract 2>&1
    blockMesh > log.blockMesh 2>&1
    snappyHexMesh -overwrite > log.snappyHexMesh 2>&1

    echo "-> 3. Δημιουργία 6DoF Ζωνών (topoSet)..."
    topoSet > log.topoSet 2>&1

    echo "-> 4. Προετοιμασία Αρχικών Συνθηκών..."
    rm -rf 0; cp -r 0.orig 0
    
    # Εξασφάλιση ότι ξεκινάει από το 0
    sed -i 's/startFrom[ \t]*latestTime;/startFrom        startTime;/g' system/controlDict
    
    decomposePar -force > log.decomposePar 2>&1

    echo "-> 5. Επίλυση overPimpleFoam (Σε $NCORES πυρήνες)..."
    mpiexec -n $NCORES overPimpleFoam -parallel > run.log 2>&1
fi

echo "-> 6. Ανασύνθεση (Reconstruct) και Έξυπνος Καθαρισμός..."
reconstructPar > log.reconstructPar 2>&1 && rm -rf processor*

touch results.foam

echo "ΕΠΙΤΥΧΙΑ! Άνοιξε το results.foam στο ParaView για να δεις το χαοτικό 3D Wake!"
"""
            with open(os.path.join(folder_name, "run_cfd.sh"), "w", encoding="utf-8", newline='\n') as f:
                f.write(bash_script.replace("\r", ""))

            print("="*60)
            print(" [ΕΠΙΤΥΧΙΑ] Το απόλυτο 6-DoF Setup ολοκληρώθηκε!")
            print("="*60 + "\n")

        except Exception as e:
            print("\n" + "!"*60)
            print(" [ΣΦΑΛΜΑ] Κάτι πήγε στραβά:")
            import traceback; traceback.print_exc()
            print("!"*60 + "\n")

    def load_state_gui(self, state):
        if not state: return
        
        try:
            print("\n[LOAD] Αναμονή για επιλογή αρχείου JSON...")
            
            # --- UNIVERSAL ROUTING ---
            if getattr(sys, 'frozen', False):
                # Αν τρέχει ως .exe
                cmd = [sys.executable, '--run-load-dialog']
            else:
                # Αν τρέχει ως .py (Προσθέτουμε το __file__ για να ξέρει ποιο script να τρέξει)
                cmd = [sys.executable, os.path.abspath(__file__), '--run-load-dialog']
                
            result = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
        except: return
        
        if result == 'CANCEL' or not os.path.exists(result):
            print("[LOAD] Ακυρώθηκε.")
            return
            
        try:
            with open(result, 'r', encoding="utf-8") as f: data = json.load(f)
            
            inputs = data["inputs"]
            self.val_water_speed = inputs["val_water_speed"]
            self.val_main_line = inputs["val_main_line"]
            self.val_winch_pitch = inputs["val_winch_pitch"]
            self.val_winch_yaw = inputs["val_winch_yaw"]
            self.val_vsurf_L = inputs["val_vsurf_L"]
            self.val_vsurf_R = inputs["val_vsurf_R"]
            self.val_gen_load = inputs["val_gen_load"]
            self.val_cyl_L = inputs["val_cyl_L"]
            self.val_cyl_r = inputs["val_cyl_r"]
            self.val_rib_frac = inputs["val_rib_frac"]
            self.val_rib_cl_max = inputs["val_rib_cl_max"]
            self.val_rib_sr_peak = inputs["val_rib_sr_peak"]
            
            self.slider_water.GetRepresentation().SetValue(self.val_water_speed)
            self.slider_main.GetRepresentation().SetValue(self.val_main_line)
            self.slider_pitch.GetRepresentation().SetValue(self.val_winch_pitch)
            self.slider_yaw.GetRepresentation().SetValue(self.val_winch_yaw)
            self.slider_spin_L.GetRepresentation().SetValue(self.val_vsurf_L)
            self.slider_spin_R.GetRepresentation().SetValue(self.val_vsurf_R)
            self.slider_load.GetRepresentation().SetValue(self.val_gen_load)
            self.slider_cyl_L.GetRepresentation().SetValue(self.val_cyl_L)
            self.slider_cyl_r.GetRepresentation().SetValue(self.val_cyl_r)
            
            k = data["kinematics"]
            self.sys_pos = np.array(k["sys_pos"], dtype=float)
            self.sys_vel = np.array(k["sys_vel"], dtype=float)
            self.sys_pitch = k["sys_pitch"]; self.sys_omega_pitch = k["sys_omega_pitch"]
            self.sys_yaw = k["sys_yaw"]; self.sys_omega_yaw = k["sys_omega_yaw"]
            
            r = data["rotors"]
            self.omega_L = r["omega_L"]; self.omega_R = r["omega_R"]
            self.rotor_angle_L = r["rotor_angle_L"]; self.rotor_angle_R = r["rotor_angle_R"]
            
            self.cyl_spins = {int(key): float(val) for key, val in r["cyl_spins"].items()}
            self.cyl_phases = {int(key): float(val) for key, val in r["cyl_phases"].items()}
            self.sim_time = data["time"]

            self.update_mass_properties()
            self.update_geometry()
            
            print(f"\n[LOAD SUCCESS] Το σύστημα τηλεμεταφέρθηκε επιτυχώς στο state του CFD!")
            print(f"Αρχείο: {result}")
            
        except Exception as e:
            print(f"\n[LOAD ERROR] Αποτυχία φόρτωσης: {e}")

    def run(self):
        self.p.show(interactive_update=True, auto_close=False, full_screen=True)
        while True:
            try:
                if not hasattr(self.p, 'render_window') or self.p.render_window is None: break
                
                if not self.is_paused:
                    self.update_geometry()
                    
                self.p.update()
                time.sleep(0.04) 
            except Exception as e:
                print("Σφάλμα στην προσομοίωση:", e); break
        try: self.p.close()
        except: pass

# --- ΑΝΕΞΑΡΤΗΤΕΣ ΔΙΕΡΓΑΣΙΕΣ ΔΙΑΛΟΓΩΝ (ΓΙΑ ΝΑ ΜΗΝ ΜΠΛΟΚΑΡΟΥΝ ΤΟ PYVISTA) ---

def isolated_load_dialog():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    file_path = filedialog.askopenfilename(title="Επιλογή State JSON", filetypes=[("JSON files", "*.json")])
    print(file_path if file_path else 'CANCEL')
    root.destroy()

def isolated_cfd_dialog(water_speed):
    import tkinter as tk
    
    root = tk.Tk()
    root.title("River-Monster: Smart CFD Exporter")
    root.geometry("500x420")
    root.attributes('-topmost', True)
    root.configure(bg='#f0f0f0')
    
    min_time = round(65.0 / water_speed, 1)
    dur_var = tk.DoubleVar(value=min_time)
    fps_var = tk.DoubleVar(value=1.0)
    size_var = tk.StringVar()
    
    def update_size(*args):
        try:
            d = float(dur_var.get()); f = float(fps_var.get())
            size_var.set("Προβλεπόμενος Χώρος Δίσκου: " + str(round((d * f) * 1.0 + 2.0, 1)) + " GB")
        except:
            size_var.set("Σφάλμα")
            
    dur_var.trace_add('write', update_size)
    fps_var.trace_add('write', update_size)
    
    tk.Label(root, text="Ταχύτητα Νερού: " + str(water_speed) + " m/s", font=('Arial', 12, 'bold'), bg='#f0f0f0').pack(pady=(15,5))
    tk.Label(root, text="Διάρκεια 6DoF Προσομοίωσης (sec):\n(Προτείνεται " + str(min_time) + "s)", font=('Arial', 10), bg='#f0f0f0').pack(pady=(10,2))
    tk.Entry(root, textvariable=dur_var, font=('Arial', 14, 'bold'), justify='center', width=10).pack(pady=5)
    tk.Label(root, text="3D Γραφικά (Frames per Second):", font=('Arial', 10), bg='#f0f0f0').pack(pady=(15,2))
    tk.Scale(root, variable=fps_var, from_=0.1, to=5.0, resolution=0.1, orient='horizontal', length=300, bg='#f0f0f0').pack(pady=5)
    tk.Label(root, textvariable=size_var, font=('Arial', 12, 'bold'), fg='red', bg='#f0f0f0').pack(pady=20)
    
    def on_submit():
        print(str(dur_var.get()) + "|" + str(fps_var.get()))
        root.destroy()
        
    def on_cancel():
        print("CANCEL")
        root.destroy()
        
    tk.Button(root, text="ΕΞΑΓΩΓΗ", command=on_submit, bg='lime', font=('Arial', 12, 'bold'), width=12).pack(side='left', padx=40, pady=10)
    tk.Button(root, text="ΑΚΥΡΟ", command=on_cancel, bg='lightgray', font=('Arial', 12), width=12).pack(side='right', padx=40, pady=10)
    
    update_size()
    root.mainloop()

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # --- ΚΡΥΦΗ ΔΡΟΜΟΛΟΓΗΣΗ ΓΙΑ ΤΟ .EXE ---
    if len(sys.argv) > 1:
        if sys.argv[1] == '--run-load-dialog':
            isolated_load_dialog()
            sys.exit(0)
        elif sys.argv[1] == '--run-cfd-dialog':
            water_v = float(sys.argv[2])
            isolated_cfd_dialog(water_v)
            sys.exit(0)
            
    # --- ΚΑΝΟΝΙΚΗ ΕΚΚΙΝΗΣΗ ΕΦΑΡΜΟΓΗΣ ---
    app = TwinMagnusHAWT_Physics()
    app.run()
