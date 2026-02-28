import pyvista as pv
import numpy as np
import time
import sys

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

def math_pts_cyl(p0, p1, r0, r1, res=24, rot=0.0):
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
    base_r = r0 + (r1 - r0) * (H / mag); Radii = base_r * 1.05 
    H_ = H[..., np.newaxis]; U_ = U[..., np.newaxis]; R_ = Radii[..., np.newaxis]
    Points = (p0 + v * H_ + R_ * np.cos(U_) * n1 + R_ * np.sin(U_) * n2)
    return Points.reshape(-1, 3), [res_w, 2, 1]

def create_solid_template(radius, height=1.0, capping=True):
    return pv.Cylinder(center=(0,0,0), direction=(0,0,1), radius=radius, height=height, resolution=48, capping=capping)
def create_arrow_template(scale=1.0):
    return pv.Arrow(start=(0,0,0), direction=(0,0,1), scale=scale, tip_length=0.25, tip_radius=0.1, shaft_radius=0.04)
def create_grid_mesh(p0, p1, r0, r1, res=24):
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
                                      smooth_shading=True, specular=0.5, name=None)
        self.mesh = self.actor.mapper.dataset
        self.user_visible = True 
        
    def set_matrix(self, matrix): self.actor.user_matrix = matrix
    
    def update_transform(self, p0, p1, scale_x=1.0, scale_y=1.0, scale_z=1.0):
        if np.any(np.isnan(p0)) or np.any(np.isnan(p1)) or np.isnan(scale_z): return
        self.actor.user_matrix = get_align_matrix(p0, p1, scale_x=scale_x, scale_y=scale_y, scale_z=scale_z)
        
    def update_arrow(self, p0, p1, max_len=15.0):
        v = p1 - p0; mag = np.linalg.norm(v)
        if mag < 0.05 or np.isnan(mag):
            self.actor.visibility = False; return
        if mag > max_len: p1 = p0 + (v / mag) * max_len
        self.actor.user_matrix = get_arrow_align_matrix(p0, p1)
        if self.user_visible: self.actor.visibility = True
            
    def set_visibility(self, visible): 
        self.user_visible = bool(visible); self.actor.visibility = self.user_visible
    def set_color(self, color): self.actor.prop.color = color

# --- MAIN APP ---
class TwinMagnusVAHT_Physics:
    def __init__(self):
        pv.global_theme.allow_empty_mesh = True
        self.p = pv.Plotter(title="River-Monster: FLIGHT DECK (STABLE PHYSICS & MASSIVE TORPEDOES)", window_size=(1600, 1000))
        self.p.set_background('white')
        
        def _on_close(*args):
            try: self.p.iren.TerminateApp()
            except: pass
            sys.exit(0)
        if hasattr(self.p, 'iren') and self.p.iren is not None:
            self.p.iren.add_observer("ExitEvent", _on_close)
            self.p.iren.add_observer("WindowCloseEvent", _on_close)
            
        self.rho = 1000.0 
        self.Cyl_L = 6.0
        self.Cyl_r = 0.5  
        
        rho_pvc = 1380.0  
        self.wall_thickness = 0.015  
        self.r_in = self.Cyl_r - self.wall_thickness
        
        Volume_PVC = np.pi * (self.Cyl_r**2 - self.r_in**2) * self.Cyl_L
        Structural_Mass = Volume_PVC * rho_pvc
        Volume_Water_Inside = np.pi * (self.r_in**2) * self.Cyl_L
        Mass_Water_Inside = Volume_Water_Inside * self.rho
        
        self.Cyl_Mass = Structural_Mass + Mass_Water_Inside  
        self.I_cyl = 0.5 * self.Cyl_Mass * (self.Cyl_r ** 2) 
        
        self.Bridge_Mass = 6000.0 
        self.System_Mass = (6 * self.Cyl_Mass) + self.Bridge_Mass 
        
        self.System_Inertia_Pitch = 1200000.0 
        self.System_Inertia_Yaw = 1800000.0 
        
        self.sys_pos = np.array([0.0, 0.0, -15.0]) 
        self.sys_vel = np.array([0.0, 0.0, 0.0])
        self.sys_pitch = 0.0; self.sys_omega_pitch = 0.0
        self.sys_yaw = 0.0; self.sys_omega_yaw = 0.0 
        
        self.anchor_pos_front = np.array([0.0, -60.0, -45.0])
        self.anchor_pos_rear = np.array([0.0, 60.0, -45.0])
        
        self.Hub_Dist_Mean = 24.0   
        self.Frame_Depth = 12.0
        self.Hub_R = 4.0        
        self.Cyl_Mid_R = self.Hub_R + (self.Cyl_L / 2.0) 
        
        self.toe_in_angle = 0.0
        
        self.Inertia_Rotor = 3.0 * (self.Cyl_Mass * (self.Cyl_Mid_R**2)) + 10000.0
        
        # --- FIXED PHYSICS ENGINE ---
        # Αποσύνδεση του οπτικού μεγέθους από την άντωση. Χρησιμοποιούμε τον "ενεργό" 
        # όγκο που διατηρεί την πλατφόρμα σε τέλεια ισορροπία με τα σκοινιά.
        Pontoon_Vol_Effective = 170.0 
        self.Buoyancy_Frame_Total_N = Pontoon_Vol_Effective * 1000.0 * 9.81
        
        self.spinning = False; self.was_spinning = False 
        
        self.show_flow_vectors = False; self.show_app_wind = False
        self.show_force_total = False; self.show_force_components = True
        self.show_stress_forces = False; self.show_detailed_stress = False 
        
        self.val_water_speed = 1.8   
        self.val_main_line = 35.0    
        
        self.val_winch_pitch = 0.0  
        self.val_winch_yaw = 0.0    
        
        self.val_spin_cyl_L = 70.0; self.val_spin_cyl_R = -70.0   
        self.val_gen_load = 25.0      
        
        self.omega_L = 0.0; self.omega_R = 0.0 
        self.rotor_angle_L = 0.0; self.rotor_angle_R = 0.0
        self.cyl_spins = {i: 0.0 for i in range(8)}
        self.cyl_phases = {i: 0.0 for i in range(8)}
        
        self.prev_thrust_L = 0.0; self.prev_thrust_R = 0.0
        self.induction_a_L = 0.0; self.induction_a_R = 0.0
        
        self.tension_FL = 0.0; self.tension_FR = 0.0; self.tension_RL = 0.0; self.tension_RR = 0.0
        self.tension_Main_Front = 0.0; self.tension_Main_Rear = 0.0
        self.beam_tension_X = 0.0; self.beam_torsion_Z = 0.0
        
        self.SR_L = 0.0; self.CL_L = 0.0; self.CD_L = 0.0
        self.SR_R = 0.0; self.CL_R = 0.0; self.CD_R = 0.0
        self.thrust_L = 0.0; self.thrust_R = 0.0; self.drive_L = 0.0; self.drive_R = 0.0
        
        self.P_mot_L = 0.0; self.P_mot_R = 0.0
        self.P_gen_mech_L = 0.0; self.P_gen_mech_R = 0.0
        self.P_gen_elec_L = 0.0; self.P_gen_elec_R = 0.0
        self.net_P_W = 0.0; self.total_P_gen_W = 0.0; self.total_P_mot_W = 0.0
        self.mech_P_gen_W = 0.0; self.hydro_power_in_W = 0.0
        
        self.frame_parts = {}; self.env_parts = {}; self.rotor_parts = []; self.dynamic_blades = []
        self.blade_vector_parts = {}; self.rotor_vector_parts = {}; self.lbl_actors = [] 
        
        self.setup_ui(); self.build_scene(); self.update_hud()
        self.p.camera.position = (45, -100, 45); self.p.camera.focal_point = (0, 0, -10); self.p.camera.up = (0, 0, 1)

    def draw_button_labels(self):
        # Νέες, κεντραρισμένες συντεταγμένες X για τις 3 στήλες των κουμπιών
        x1 = 450; x2 = 800; x3 = 1150; row1_y = 120; row2_y = 75; row3_y = 30
        self.lbl_actors = [
            self.p.add_text("START SIM", position=(x1 + 40, row1_y+5), color='black', font_size=12),
            self.p.add_text("AERO FLOW VECTORS", position=(x2 + 40, row1_y+5), color='black', font_size=12),
            self.p.add_text("ROTOR STRESS", position=(x3 + 40, row1_y+5), color='black', font_size=12),
            self.p.add_text("APPARENT WATER", position=(x1 + 40, row2_y+5), color='black', font_size=12),
            self.p.add_text("TOTAL FORCE", position=(x2 + 40, row2_y+5), color='black', font_size=12),
            self.p.add_text("FORCE COMPONENTS", position=(x3 + 40, row2_y+5), color='black', font_size=12),
            self.p.add_text("DETAILED NODE STRESS", position=(x2 + 40, row3_y+5), color='black', font_size=13, font='courier')
        ]

    def setup_ui(self):
        self.p.add_slider_widget(self.set_water, [0, 5.0], title="Water Speed (m/s)", value=self.val_water_speed, pointa=(0.02, 0.93), pointb=(0.18, 0.93), style='modern')
        self.p.add_slider_widget(self.set_main_line, [10, 60], title="Main Lines Length (m)", value=self.val_main_line, pointa=(0.02, 0.81), pointb=(0.18, 0.81), style='modern')
        
        self.p.add_slider_widget(self.set_winch_pitch, [-10.0, 10.0], title="Vertical Pitch Diff (m)", value=self.val_winch_pitch, pointa=(0.02, 0.69), pointb=(0.18, 0.69), style='modern')
        self.p.add_slider_widget(self.set_winch_yaw, [-10.0, 10.0], title="Horizontal Yaw Diff (m)", value=self.val_winch_yaw, pointa=(0.02, 0.57), pointb=(0.18, 0.57), style='modern')
        
        self.p.add_slider_widget(self.set_spin_L, [-400, 400], title="Left Spin (RPM)", value=self.val_spin_cyl_L, pointa=(0.02, 0.45), pointb=(0.18, 0.45), style='modern')
        self.p.add_slider_widget(self.set_spin_R, [-400, 400], title="Right Spin (RPM)", value=self.val_spin_cyl_R, pointa=(0.02, 0.33), pointb=(0.18, 0.33), style='modern')
        self.p.add_slider_widget(self.set_load, [0, 100], title="Generator Load (%)", value=self.val_gen_load, pointa=(0.02, 0.21), pointb=(0.18, 0.21), style='modern')
        
        # Ίδιες συντεταγμένες με τα labels για να "κουμπώσουν" σωστά τα κουτιά δίπλα στα γράμματα
        x1 = 450; x2 = 800; x3 = 1150; row1_y = 120; row2_y = 75; row3_y = 30
        
        self.p.add_checkbox_button_widget(self.toggle_spin, value=self.spinning, position=(x1, row1_y), size=30, color_on='green', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_flow, value=self.show_flow_vectors, position=(x2, row1_y), size=30, color_on='lime', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_stress, value=self.show_stress_forces, position=(x3, row1_y), size=30, color_on='magenta', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_app_wind, value=self.show_app_wind, position=(x1, row2_y), size=30, color_on='orange', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_force_total, value=self.show_force_total, position=(x2, row2_y), size=30, color_on='purple', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_force_comp, value=self.show_force_components, position=(x3, row2_y), size=30, color_on='blue', color_off='grey')
        self.p.add_checkbox_button_widget(self.toggle_detailed, value=self.show_detailed_stress, position=(x2, row3_y), size=30, color_on='cyan', color_off='grey')
        self.draw_button_labels()

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

        hud = f"""
=====================================================================================================
 FLIGHT DECK / TELEMETRY OVERVIEW   |   STABLE PHYSICS & MASSIVE TORPEDOES
=====================================================================================================
 [ KINEMATICS & ENVIRONMENT ]             [ STRUCTURAL STRESS & MOORING (kN) ]
  Water Speed  : {w_spd:>6.2f} m/s                  Main Lines F/R : {t_mf:>6.1f} / {t_mr:>6.1f} kN
  Sys Depth    : {dpth:>6.2f} m                       Bridle FL/FR   : {t_fl:>8.1f} / {t_fr:>8.1f} kN
  Sys Pitch    : {ptch:>6.2f} deg                       Bridle RL/RR   : {t_rl:>8.1f} / {t_rr:>8.1f} kN
  Sys Yaw      : {yw:>6.2f} deg                       Rear Beam      : {t_beam:>8.1f} kN (Tension)
  Frame Buoy   : {fb:>6.1f} t                       Beam Torsion   : {t_torsion:>8.1f} kNm
                                          Rotor Thrust   : L:{thr_l:>6.1f} | R:{thr_r:>6.1f} kN

 [ PORT ROTOR (LEFT) TELEMETRY ]          [ STARBOARD ROTOR (RIGHT) TELEMETRY ]
  Cyl Spin     : {rpm_cyl_l:>6.0f} RPM                  Cyl Spin     : {rpm_cyl_r:>6.0f} RPM
  Hub Orbit    : {rpm_hub_l:>6.1f} RPM                  Hub Orbit    : {rpm_hub_r:>6.1f} RPM
  Spin Ratio   : {sr_l:>6.2f} (a_vperp)             Spin Ratio   : {sr_r:>6.2f} (a_vperp)
  Aero Polars  : CL={cl_l:>4.2f} | CD={cd_l:>4.2f}          Aero Polars  : CL={cl_r:>4.2f} | CD={cd_r:>4.2f}
  Drive Force  : {drv_l:>6.1f} kN                   Drive Force  : {drv_r:>6.1f} kN

 [ POWER SYSTEMS & EFFICIENCY (Generator Eff: 85%) ]
  Hydro Power In (Avail) : {hydro_in:>8.2f} kW
  PORT (L)  : Mot Draw: {pm_l:>7.2f} kW | Gen Elec: +{pge_l:>7.2f} kW
  STBD (R)  : Mot Draw: {pm_r:>7.2f} kW | Gen Elec: +{pge_r:>7.2f} kW
-----------------------------------------------------------------------------------------------------
  SYSTEM NET ELECTRICAL POWER :  {net:>+8.2f} kW
=====================================================================================================
"""
        width, height = self.p.window_size
        self.p.add_text(hud, position=(width - 780, height - 370), color='black', font_size=7, font='courier', name='hud_block')

    def set_water(self, val): self.val_water_speed = val
    def set_main_line(self, val): self.val_main_line = val
    def set_winch_pitch(self, val): self.val_winch_pitch = val
    def set_winch_yaw(self, val): self.val_winch_yaw = val
    def set_spin_L(self, val): self.val_spin_cyl_L = val
    def set_spin_R(self, val): self.val_spin_cyl_R = val
    def set_load(self, val): self.val_gen_load = val

    def toggle_spin(self, state): self.spinning = state
    def toggle_flow(self, state): self.show_flow_vectors = state; self.update_geometry()
    def toggle_app_wind(self, state): self.show_app_wind = state; self.update_geometry()
    def toggle_force_total(self, state): self.show_force_total = state; self.update_geometry()
    def toggle_force_comp(self, state): self.show_force_components = state; self.update_geometry()
    def toggle_stress(self, state): self.show_stress_forces = state; self.update_geometry()
    def toggle_detailed(self, state): self.show_detailed_stress = state; self.update_geometry()

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
        
        self.env_parts['Force_Frame_L'] = ScenePart(self.p, create_arrow_template(scale=1.5), 'orange')
        self.env_parts['Force_Frame_R'] = ScenePart(self.p, create_arrow_template(scale=1.5), 'orange')
        
        for dp in ['Tension_FL', 'Tension_FR', 'Tension_RL', 'Tension_RR', 'Tension_Main_Front', 'Tension_Main_Rear']:
            self.env_parts[dp] = ScenePart(self.p, create_arrow_template(scale=1.0), 'cyan')
            self.env_parts[dp].set_visibility(False)
            
        self.env_parts['Beam_Tension_L'] = ScenePart(self.p, create_arrow_template(scale=1.5), 'yellow')
        self.env_parts['Beam_Tension_R'] = ScenePart(self.p, create_arrow_template(scale=1.5), 'yellow')

        rotors_info = [('Left', 0), ('Right', 1)]
        blade_idx = 0 
        for tag, rotor_id in rotors_info:
            rotor_parts = {}
            rotor_parts['Hub'] = ScenePart(self.p, create_solid_template(self.Hub_R, height=0.6, capping=False), 'silver')
            rotor_parts['Hub_Ext'] = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.15, 0.15), 'silver') 
            rotor_parts['Pontoon_Rim'] = ScenePart(self.p, create_solid_template(3.05, height=0.2, capping=False), 'darkgrey')
            self.rotor_parts.append(rotor_parts)
            self.rotor_vector_parts[rotor_id] = ScenePart(self.p, create_arrow_template(scale=4.0), 'magenta')
            self.rotor_vector_parts[rotor_id].set_visibility(False)
            for deg in [0, 120, 240]:
                strut = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.1, 0.1), 'black')
                motor_bell_inner = ScenePart(self.p, create_solid_template(self.Cyl_r * 0.8, height=0.5), 'black')
                blade_cyl = ScenePart(self.p, create_grid_mesh([0,0,-self.Cyl_L/2], [0,0,self.Cyl_L/2], self.Cyl_r, self.Cyl_r, res=30), 'blue', opacity=0.6)
                tape = ScenePart(self.p, create_strip_mesh_init([0,0,-self.Cyl_L/2], [0,0,self.Cyl_L/2], self.Cyl_r, self.Cyl_r, 0.0), 'black')
                
                plate_radius = self.Cyl_r * 2.5  
                plate_top = ScenePart(self.p, create_solid_template(plate_radius, height=0.05), 'darkgrey')
                plate_bot = ScenePart(self.p, create_solid_template(plate_radius, height=0.05), 'darkgrey')
                wire = ScenePart(self.p, create_grid_mesh([0,0,0], [1,0,0], 0.03, 0.03), 'black') 
                vectors = {}
                for key, col, scale in [('flow_A','lime',1), ('flow_B','red',1), ('air_app','orange',1), 
                                 ('force_total','purple',1), ('force_drive','blue',1), ('force_axial','red',1),
                                 ('stress_tip', 'magenta', 1.5), ('stress_hub', 'cyan', 1.5)]:
                    vectors[key] = ScenePart(self.p, create_arrow_template(scale=scale), col)
                    vectors[key].set_visibility(False)
                self.dynamic_blades.append({
                    'tag': tag, 'rotor_id': rotor_id, 'base_angle': np.radians(deg), 
                    'strut': strut, 'bell_inner': motor_bell_inner,
                    'plate_top': plate_top, 'plate_bot': plate_bot,
                    'blade_cyl': blade_cyl, 'tape': tape, 'wire': wire, 'id': blade_idx
                })
                self.blade_vector_parts[blade_idx] = vectors
                blade_idx += 1
        self.update_geometry()

    def update_geometry(self):
        dt = 0.04 
        self.P_mot_L = 0.0; self.P_mot_R = 0.0
        self.tension_FL = 0.0; self.tension_FR = 0.0; self.tension_RL = 0.0; self.tension_RR = 0.0
        self.tension_Main_Front = 0.0; self.tension_Main_Rear = 0.0
        
        Cf_water = 0.005
        Area_cyl_out = 2 * np.pi * self.Cyl_r * self.Cyl_L
        Area_cyl_in = 2 * np.pi * self.r_in * self.Cyl_L
        Motor_Efficiency = 0.85
        
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
        
        # --- OΠΤΙΚΟΣ ΔΙΠΛΑΣΙΑΣΜΟΣ ΜΗΚΟΥΣ ---
        # scale_z=1.0 δημιουργεί μήκος 24m, ενώ το scale_x=3.0 και scale_y=3.0 τα κάνει χοντρά.
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
            r_z = rotor_z_L_local if i == 0 else rotor_z_R_local
            r_x = rotor_x_L_local if i == 0 else rotor_x_R_local
            r_axis_glob = rot_axis_L_global if i == 0 else rot_axis_R_global
            
            m_hub = np.eye(4); m_hub[0:3, 0] = R_mat.dot(r_x); m_hub[0:3, 1] = R_mat.dot(rotor_y_local)
            m_hub[0:3, 2] = R_mat.dot(r_z); m_hub[0:3, 3] = hub_global
            parts['Hub'].set_matrix(m_hub)

            m_rim = np.copy(m_hub)
            parts['Pontoon_Rim'].set_matrix(m_rim)

            p_ext = hub_global + r_axis_glob * 2.5
            parts['Hub_Ext'].mesh.points, _ = math_pts_cyl(hub_global, p_ext, 0.15, 0.15)

        A_swept = np.pi * ((self.Hub_R + self.Cyl_L)**2 - self.Hub_R**2) 
        dyn_pressure = 0.5 * self.rho * max(self.val_water_speed**2, 0.01)
        self.hydro_power_in_W = 0.5 * self.rho * (A_swept * 2) * (self.val_water_speed**3)

        C_T_L = np.clip(self.prev_thrust_L / (dyn_pressure * A_swept + 0.001), 0.0, 0.8) 
        C_T_R = np.clip(self.prev_thrust_R / (dyn_pressure * A_swept + 0.001), 0.0, 0.8)
        self.induction_a_L = 0.5 * (1.0 - np.sqrt(1.0 - C_T_L))
        self.induction_a_R = 0.5 * (1.0 - np.sqrt(1.0 - C_T_R))

        v_water_global = np.array([0.0, self.val_water_speed, 0.0])
        Sys_Net_Force = np.array([0.0, 0.0, -self.System_Mass * 9.81]) 
        Sys_Net_Torque = np.array([0.0, 0.0, 0.0]) 
        
        Buoy_per_pontoon = self.Buoyancy_Frame_Total_N / 2.0
        F_buoy = np.array([0.0, 0.0, Buoy_per_pontoon])
        Sys_Net_Force += (2 * F_buoy)
        Sys_Net_Torque += np.cross(mid_L - self.sys_pos, F_buoy)
        Sys_Net_Torque += np.cross(mid_R - self.sys_pos, F_buoy)
        
        m_anc_f = np.eye(4); m_anc_f[0:3, 3] = self.anchor_pos_front; self.env_parts['Anchor_Front'].set_matrix(m_anc_f)
        m_anc_r = np.eye(4); m_anc_r[0:3, 3] = self.anchor_pos_rear; self.env_parts['Anchor_Rear'].set_matrix(m_anc_r)
        
        vec_anc_f_to_center = self.sys_pos - self.anchor_pos_front
        dist_f = np.linalg.norm(vec_anc_f_to_center)
        dir_f = vec_anc_f_to_center / max(dist_f, 0.001)
        knot_pos_front = self.anchor_pos_front + dir_f * self.val_main_line
        
        vec_anc_r_to_center = self.sys_pos - self.anchor_pos_rear
        dist_r = np.linalg.norm(vec_anc_r_to_center)
        dir_r = vec_anc_r_to_center / max(dist_r, 0.001)
        knot_pos_rear = self.anchor_pos_rear + dir_r * self.val_main_line
        
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
        
        scale_detailed = 0.00005 
        for corner_pos, winch_len, part_name, tension_name, k_pos in winches:
            vec_corner_to_knot = k_pos - corner_pos
            dist_ck = np.linalg.norm(vec_corner_to_knot)
            stretch_ck = dist_ck - winch_len
            if stretch_ck > 0:
                F_pull_mag = stretch_ck * 200000.0
                if part_name == 'Bridle_FL': self.tension_FL = F_pull_mag
                elif part_name == 'Bridle_FR': self.tension_FR = F_pull_mag
                elif part_name == 'Bridle_RL': self.tension_RL = F_pull_mag
                elif part_name == 'Bridle_RR': self.tension_RR = F_pull_mag
                
                F_pull_vec = F_pull_mag * (vec_corner_to_knot / max(dist_ck, 0.001))
                Sys_Net_Force += F_pull_vec
                Sys_Net_Torque += np.cross(corner_pos - self.sys_pos, F_pull_vec)
                self.env_parts[tension_name].update_arrow(corner_pos, corner_pos + F_pull_vec * scale_detailed, 12.0)
                self.env_parts[tension_name].set_visibility(self.show_detailed_stress and self.spinning)
            else: self.env_parts[tension_name].set_visibility(False)
            self.env_parts[part_name].mesh.points, _ = math_pts_cyl(corner_pos, k_pos, 0.04, 0.04)
        
        stretch_anc_f = dist_f - self.val_main_line
        if stretch_anc_f > 0:
            self.tension_Main_Front = stretch_anc_f * 500000.0
            F_anc_vec_f = self.tension_Main_Front * dir_f
            self.env_parts['Tension_Main_Front'].update_arrow(knot_pos_front, knot_pos_front - F_anc_vec_f * scale_detailed, 15.0)
            self.env_parts['Tension_Main_Front'].set_visibility(self.show_detailed_stress and self.spinning)
        else: self.env_parts['Tension_Main_Front'].set_visibility(False)
            
        stretch_anc_r = dist_r - self.val_main_line
        if stretch_anc_r > 0:
            self.tension_Main_Rear = stretch_anc_r * 500000.0
            F_anc_vec_r = self.tension_Main_Rear * dir_r
            self.env_parts['Tension_Main_Rear'].update_arrow(knot_pos_rear, knot_pos_rear - F_anc_vec_r * scale_detailed, 15.0)
            self.env_parts['Tension_Main_Rear'].set_visibility(self.show_detailed_stress and self.spinning)
        else: self.env_parts['Tension_Main_Rear'].set_visibility(False)
            
        m_kf = np.eye(4); m_kf[0:3, 3] = knot_pos_front; self.env_parts['Knot_Front'].set_matrix(m_kf)
        m_kr = np.eye(4); m_kr[0:3, 3] = knot_pos_rear; self.env_parts['Knot_Rear'].set_matrix(m_kr)
        self.env_parts['Main_Line_Front'].mesh.points, _ = math_pts_cyl(self.anchor_pos_front, knot_pos_front, 0.1, 0.1)
        self.env_parts['Main_Line_Rear'].mesh.points, _ = math_pts_cyl(self.anchor_pos_rear, knot_pos_rear, 0.1, 0.1)

        torque_net_L = 0.0; torque_net_R = 0.0; frame_thrust_L = 0.0; frame_thrust_R = 0.0
        net_rotor_force = {0: np.array([0.,0.,0.]), 1: np.array([0.,0.,0.])}
            
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
            
            item['blade_cyl'].mesh.points, _ = math_pts_cyl(p_cyl_base, p_tip_center, self.Cyl_r, self.Cyl_r, 30, 0.0)
            item['tape'].mesh.points, _ = math_pts_strip(p_cyl_base, p_tip_center, self.Cyl_r, self.Cyl_r, 0.0, self.cyl_phases[b_id])
            
            a_local = self.induction_a_L if is_left else self.induction_a_R
            wake_multiplier = 1.0 - a_local
            local_v_water = v_water_global * max(wake_multiplier, 0.1)

            base_rpm = self.val_spin_cyl_L if is_left else self.val_spin_cyl_R
            target_rpm = base_rpm 
                
            old_cyl_rpm = self.cyl_spins[b_id]
            if self.spinning: self.cyl_spins[b_id] += (target_rpm - self.cyl_spins[b_id]) * dt * 2.0
            else: self.cyl_spins[b_id] *= 0.9
            
            cyl_rpm = self.cyl_spins[b_id]; cyl_omega = cyl_rpm * 0.1047
            alpha_cyl = (cyl_omega - old_cyl_rpm * 0.1047) / dt
            if self.spinning: self.cyl_phases[b_id] += cyl_omega * dt
            
            v_surface_out = abs(cyl_omega * self.Cyl_r)
            v_surface_in = abs(cyl_omega * self.r_in)
            
            P_skin_out = self.rho * Cf_water * Area_cyl_out * (v_surface_out**3)
            P_skin_in = self.rho * Cf_water * Area_cyl_in * (v_surface_in**3)
            
            P_skin_friction_W = P_skin_out + P_skin_in
            P_pump_drag_W = 0.0  
            P_inertia_W = self.I_cyl * alpha_cyl * cyl_omega 
            P_friction_W = P_skin_friction_W + P_pump_drag_W
            
            N_strips = 10; dr_strip = cyl_length / N_strips
            total_force = np.array([0.0, 0.0, 0.0]); total_torque_blade = 0.0; total_thrust_comp = 0.0; P_crossflow_W = 0.0
            mid_strip_idx = N_strips // 2; v_app_mag = 0.01; safe_v_app = 0.01
            v_app_dir = np.array([0, 1, 0]); spin_axis = cyl_vec_dir
            
            AR_geometric = self.Cyl_L / (2.0 * self.Cyl_r)
            e_oswald = 0.65; area_proj_strip = dr_strip * (self.Cyl_r * 2.0)
            
            for step in range(N_strips):
                p_local = p_cyl_base + cyl_vec_dir * (step + 0.5) * dr_strip
                vec_hub_to_p = p_local - hub_global; r_local_flat = np.dot(vec_hub_to_p, rad_dir_global_flat)
                v_blade_local = np.cross(rot_axis_global * rotor_omega, rad_dir_global_flat * r_local_flat) + self.sys_vel
                app_vec_local = local_v_water - v_blade_local
                v_app_spanwise = np.dot(app_vec_local, cyl_vec_dir) * cyl_vec_dir
                v_app_perp = app_vec_local - v_app_spanwise
                v_app_perp_mag = np.linalg.norm(v_app_perp); safe_v_app_perp = max(v_app_perp_mag, 0.01)
                v_app_perp_dir = v_app_perp / safe_v_app_perp
                
                if step == mid_strip_idx:
                    v_app_mag = np.linalg.norm(app_vec_local); safe_v_app = max(v_app_mag, 0.01)  
                    v_app_dir = app_vec_local / safe_v_app
                    spin_axis = cyl_vec_dir * np.sign(cyl_rpm) if cyl_rpm != 0 else cyl_vec_dir
                
                if self.spinning and v_app_perp_mag > 0.05:
                    spin_axis_local = cyl_vec_dir * np.sign(cyl_rpm) if cyl_rpm != 0 else cyl_vec_dir
                    v_surface_local = abs(cyl_omega * self.Cyl_r)
                    spin_ratio = v_surface_local / safe_v_app_perp
                    
                    if spin_ratio < 1.5: CL = 1.8 * spin_ratio
                    else: CL = 2.7 + 0.5 * np.tanh(spin_ratio - 1.5) 
                        
                    AR_effective = (AR_geometric * 2.0) * (1.0 + np.exp(-0.5 * spin_ratio))
                    CD_profile = 1.1 + 0.15 * (spin_ratio**2) 
                    CD_induced = (CL**2) / (np.pi * AR_effective * e_oswald)
                    CD_total = CD_profile + CD_induced
                    
                    if b_id == 0 and step == mid_strip_idx: self.SR_L = spin_ratio; self.CL_L = CL; self.CD_L = CD_total
                    elif b_id == 3 and step == mid_strip_idx: self.SR_R = spin_ratio; self.CL_R = CL; self.CD_R = CD_total
                        
                    mag_lift_dir = np.cross(v_app_perp_dir, spin_axis_local)
                    if np.linalg.norm(mag_lift_dir) > 0.001: mag_lift_dir /= np.linalg.norm(mag_lift_dir)
                    dF_magnus = mag_lift_dir * (0.5 * self.rho * (safe_v_app_perp**2) * area_proj_strip * CL)
                    dF_drag = v_app_perp_dir * (0.5 * self.rho * (safe_v_app_perp**2) * area_proj_strip * CD_total)
                    dF_total = dF_magnus + dF_drag
                    
                    C_m = 0.015 * spin_ratio
                        
                    dTorque_crossflow = C_m * 0.5 * self.rho * (safe_v_app_perp**2) * area_proj_strip * (2.0 * self.Cyl_r)
                    dP_crossflow_W = dTorque_crossflow * abs(cyl_omega); P_crossflow_W += dP_crossflow_W
                    
                    total_force += dF_total; total_torque_blade += np.dot(dF_total, sweep_dir_cyl) * r_local_flat
                    total_thrust_comp += np.dot(dF_total, rot_axis_global); Sys_Net_Torque += np.cross(p_local - self.sys_pos, dF_total)

            Sys_Net_Force += total_force
            force_drive_vec = sweep_dir_cyl * np.dot(total_force, sweep_dir_cyl)
            force_axial_vec = rot_axis_global * np.dot(total_force, rot_axis_global)
            
            if is_left: torque_net_L += total_torque_blade; frame_thrust_L += total_thrust_comp
            else: torque_net_R += total_torque_blade; frame_thrust_R += total_thrust_comp
                
            P_mech_W = P_friction_W + P_inertia_W + P_crossflow_W
            if self.spinning and P_mech_W > 0:
                if is_left: self.P_mot_L += P_mech_W / Motor_Efficiency
                else: self.P_mot_R += P_mech_W / Motor_Efficiency

            F_centrifugal_mag = self.Cyl_Mass * (rotor_omega**2) * self.Cyl_Mid_R
            F_centrifugal_vec = rad_dir_global_flat * F_centrifugal_mag
            total_arm_load_vec = force_axial_vec + F_centrifugal_vec 
            net_rotor_force[r_id] += (total_force + F_centrifugal_vec)

            # --- VISUALIZATIONS ---
            mid = hub_global + rad_dir_global_flat * self.Cyl_Mid_R; vecs = self.blade_vector_parts[b_id]
            v_scale_force = 0.00003; v_scale_stress = 0.00003
            
            if v_app_mag > 0.1 and self.show_flow_vectors and self.spinning and abs(cyl_rpm) > 1.0:
                 cross = np.cross(v_app_dir, spin_axis); offset_dist = self.Cyl_r * 1.8 
                 pR = mid + cross * offset_dist; pL = mid - cross * offset_dist
                 rot_vec = rad_dir_global_flat * cyl_omega
                 vR = safe_v_app + np.dot(np.cross(rot_vec, pR-mid), v_app_dir)
                 vL = safe_v_app + np.dot(np.cross(rot_vec, pL-mid), v_app_dir)
                 scaleR = max(0.1, abs(vR)*0.1); scaleL = max(0.1, abs(vL)*0.1)
                 vecs['flow_A'].update_arrow(pR, pR+v_app_dir*scaleR, 2.5); vecs['flow_A'].set_color('lime' if vR>vL else 'red'); vecs['flow_A'].set_visibility(True)
                 vecs['flow_B'].update_arrow(pL, pL+v_app_dir*scaleL, 2.5); vecs['flow_B'].set_color('lime' if vL>vR else 'red'); vecs['flow_B'].set_visibility(True)
            else: vecs['flow_A'].set_visibility(False); vecs['flow_B'].set_visibility(False)

            scale_app = max(0.2, v_app_mag * 0.5)
            vecs['air_app'].update_arrow(mid - v_app_dir * min(scale_app, 3.0), mid, 4.0)
            vecs['air_app'].set_visibility(self.show_app_wind and v_app_mag > 0.1)
            vecs['force_total'].update_arrow(mid, mid + total_force * v_scale_force, 5.0); vecs['force_total'].set_visibility(self.show_force_total and self.spinning)
            vecs['force_drive'].update_arrow(mid, mid + force_drive_vec * v_scale_force, 5.0); vecs['force_drive'].set_visibility(self.show_force_components and self.spinning)
            vecs['force_axial'].update_arrow(mid, mid + force_axial_vec * v_scale_force, 5.0); vecs['force_axial'].set_visibility(self.show_force_components and self.spinning)
            vecs['stress_tip'].update_arrow(p_tip_center, p_tip_center + total_arm_load_vec * v_scale_stress, 7.0); vecs['stress_tip'].set_visibility(self.show_stress_forces and self.spinning)
            p_ext = hub_global + rot_axis_global * 2.5
            vecs['stress_hub'].update_arrow(p_ext, p_ext - total_arm_load_vec * v_scale_stress, 7.0); vecs['stress_hub'].set_visibility(self.show_stress_forces and self.spinning)

        self.thrust_L = frame_thrust_L; self.thrust_R = frame_thrust_R
        self.drive_L = torque_net_L / self.Cyl_Mid_R if self.Cyl_Mid_R > 0 else 0; self.drive_R = torque_net_R / self.Cyl_Mid_R if self.Cyl_Mid_R > 0 else 0
        self.prev_thrust_L = frame_thrust_L; self.prev_thrust_R = frame_thrust_R
        self.beam_tension_X = max(abs(net_rotor_force[0][0]), abs(net_rotor_force[1][0]))
        self.beam_torsion_Z = abs(net_rotor_force[0][0] - net_rotor_force[1][0]) * (self.Hub_Dist_Mean / 2.0)

        scale_frame_stress = 0.0002 
        self.env_parts['Force_Frame_L'].update_arrow(m_RL, m_RL + net_rotor_force[0] * scale_frame_stress, 15.0); self.env_parts['Force_Frame_L'].set_visibility(self.show_stress_forces and self.spinning)
        self.env_parts['Force_Frame_R'].update_arrow(m_RR, m_RR + net_rotor_force[1] * scale_frame_stress, 15.0); self.env_parts['Force_Frame_R'].set_visibility(self.show_stress_forces and self.spinning)
        
        if self.show_detailed_stress and self.spinning:
            mid_rear = (m_RL + m_RR) / 2.0; vec_L_dir = R_mat.dot(np.array([-1, 0, 0])); vec_R_dir = R_mat.dot(np.array([1, 0, 0]))
            self.env_parts['Beam_Tension_L'].update_arrow(mid_rear, mid_rear + vec_L_dir * self.beam_tension_X * scale_detailed * 4, 12.0)
            self.env_parts['Beam_Tension_R'].update_arrow(mid_rear, mid_rear + vec_R_dir * self.beam_tension_X * scale_detailed * 4, 12.0)
            self.env_parts['Beam_Tension_L'].set_visibility(True); self.env_parts['Beam_Tension_R'].set_visibility(True)
        else: self.env_parts['Beam_Tension_L'].set_visibility(False); self.env_parts['Beam_Tension_R'].set_visibility(False)

        Area_surge = 12.0; Area_heave_pitch = 100.0; CD_frame = 1.1 # Αυξημένη απόσβεση λόγω μεγαλύτερου μήκους
        F_hydro_drag = -0.5 * self.rho * CD_frame * Area_surge * self.sys_vel * np.abs(self.sys_vel)
        lever_arm = self.Frame_Depth / 2.0; v_pitch_ends = self.sys_omega_pitch * lever_arm
        T_hydro_damp_pitch = -2.0 * (0.5 * self.rho * CD_frame * Area_heave_pitch * v_pitch_ends * np.abs(v_pitch_ends)) * lever_arm
        v_yaw_ends = self.sys_omega_yaw * (self.Hub_Dist_Mean / 2.0)
        T_hydro_damp_yaw = -2.0 * (0.5 * self.rho * CD_frame * Area_surge * v_yaw_ends * np.abs(v_yaw_ends)) * (self.Hub_Dist_Mean / 2.0)

        accel = (Sys_Net_Force + F_hydro_drag) / self.System_Mass
        self.sys_vel += accel * dt; self.sys_pos += self.sys_vel * dt; self.sys_pos[2] = min(self.sys_pos[2], -1.0) 
        alpha_pitch = (Sys_Net_Torque[0] + T_hydro_damp_pitch) / self.System_Inertia_Pitch; self.sys_omega_pitch += alpha_pitch * dt; self.sys_pitch += self.sys_omega_pitch * dt
        alpha_yaw = (Sys_Net_Torque[2] + T_hydro_damp_yaw) / self.System_Inertia_Yaw; self.sys_omega_yaw += alpha_yaw * dt; self.sys_yaw += self.sys_omega_yaw * dt

        if self.spinning:
            gen_max_torque = 500000.0; app_torque = gen_max_torque * (self.val_gen_load / 100.0)
            frame_drag_L = 40000.0 * self.omega_L * abs(self.omega_L); frame_drag_R = 40000.0 * self.omega_R * abs(self.omega_R)
            brake_L = app_torque * np.sign(self.omega_L) if abs(self.omega_L) > 0.05 else 0.0
            brake_R = app_torque * np.sign(self.omega_R) if abs(self.omega_R) > 0.05 else 0.0
            
            self.P_gen_mech_L = abs(brake_L * self.omega_L); self.P_gen_mech_R = abs(brake_R * self.omega_R)
            self.P_gen_elec_L = self.P_gen_mech_L * 0.85; self.P_gen_elec_R = self.P_gen_mech_R * 0.85
            
            net_L = torque_net_L - frame_drag_L - brake_L; net_R = torque_net_R - frame_drag_R - brake_R
            alpha_L = net_L / self.Inertia_Rotor; alpha_R = net_R / self.Inertia_Rotor
            
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
        self.update_hud()

    def run(self):
        self.p.show(interactive_update=True, auto_close=False, full_screen=True)
        while True:
            try:
                if not hasattr(self.p, 'render_window') or self.p.render_window is None: break
                needs_update = False
                if self.spinning or self.was_spinning: needs_update = True
                self.was_spinning = self.spinning
                if needs_update or True: self.update_geometry()
                self.p.update(); time.sleep(0.04) 
            except Exception as e:
                print("Σφάλμα στην προσομοίωση:", e); break
        try: self.p.close()
        except: pass

if __name__ == "__main__":
    app = TwinMagnusVAHT_Physics()
    app.run()