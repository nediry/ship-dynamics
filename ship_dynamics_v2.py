"""
Created on Fri Jun 25 17:13:29 2021
@author: nedir ymamov
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import dataframe_image as dfi

class ShipDynamics:
    def __init__(self, boy, genislik, draft, dalga_genligi, blok_katsayisi, dalga_acisi, dalga_katsayisi, yogunluk, yer_cekimi):
        self.boy = boy
        self.genislik = genislik
        self.draft = draft
        self.dalga_genligi = dalga_genligi
        self.blok_katsayisi = blok_katsayisi
        self.dalga_acisi = np.deg2rad(dalga_acisi)
        self.dalga_katsayisi = dalga_katsayisi
        self.yogunluk = yogunluk
        self.yer_cekimi = yer_cekimi
        self.velocity = self.dalga_katsayisi * np.sqrt(self.yer_cekimi * self.boy)
        self.a33 = []
        self.b33 = []
        self.F3R = []
        self.F3I = []
        self.F5R = []
        self.F5I = []
    

    def added_mass(self, we):
        a33_x = [.389, .5, .621, .75, 1, 1.25, 1.5, 1.75, 2]
        a33_y = [5, 4.117, 3.494, 3.083, 2.741, 2.705, 2.848, 3.046, 3.239]
        x = we * np.sqrt(self.genislik / (2 * self.yer_cekimi))
        a33 = np.interp(x, a33_x, a33_y) * self.genislik * self.draft * self.yogunluk
        return a33
    

    def damping(self, we):
        b33_x = [0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2]
        b33_y = [0, 1.567, 2.162, 2.202, 1.919, 1.471, .992, .6, .332]
        x = we * np.sqrt(self.genislik / (2 * self.yer_cekimi))
        b33 = np.interp(x, b33_x, b33_y) * self.yogunluk * self.genislik * self.draft / (np.sqrt(self.genislik / (2 * self.yer_cekimi)))
        return b33
    

    def calculate_heave_pitch(self, dalga_frekansi, dalga_katsayisi):
        we = dalga_frekansi - self.velocity * dalga_katsayisi * np.cos(self.dalga_acisi)
        a33 = self.added_mass(we)
        b33 = self.damping(we)
        
        # HEAVE HAREKETİ FROUDE-KRYLOV KUVVETİ
        F3FK = 2 * self.yogunluk * self.yer_cekimi * self.dalga_genligi * self.genislik \
             * np.exp(-dalga_katsayisi * self.draft) * np.sin(dalga_katsayisi * self.boy / 2) / dalga_katsayisi
        
        # HEAVE KAREKETİ DİFRAKSİYON KUVVETİ
        F3D = 2 * self.dalga_genligi * np.exp(-dalga_katsayisi * self.draft) * np.sin(dalga_katsayisi * self.boy / 2) \
            * (-a33 * self.yer_cekimi + b33 * dalga_frekansi * 1j / self.dalga_genligi) 
            

        F3 = F3FK + F3D
        
        # PİTCH HAREKETİ FROUDE-KRYLOV KUVVETİ
        F5FK = self.yogunluk * self.yer_cekimi * self.dalga_genligi * self.genislik \
             * np.exp(-dalga_katsayisi * self.draft) * (2 * np.sin(dalga_katsayisi * self.boy / 2) \
             - self.boy * dalga_katsayisi * np.cos(dalga_katsayisi * self.boy / 2)) * 1j / dalga_katsayisi**2
        
        # PİTCH HAREKETİ DİFRAKSİYON KUVVETİ
        F5D = -(a33 * we * 1j + b33) * self.dalga_genligi * dalga_frekansi * (2 * np.sin(dalga_katsayisi * self.boy / 2) - self.boy \
            * dalga_katsayisi * np.cos(dalga_katsayisi * self.boy / 2)) * np.exp(-dalga_katsayisi * self.draft) / dalga_katsayisi**2
        
        F5 = F5FK + F5D
        M = self.boy * self.genislik * self.draft * self.blok_katsayisi * self.yogunluk

        # JİRASYON YARIÇAPI
        kyy = .25 * self.boy
        I = M * kyy**2
        
        A33 = a33 * self.boy 
        B33 = b33 * self.boy
        C33 = self.yogunluk * self.yer_cekimi * self.genislik * self.boy
        
        A35 = - B33 * self.velocity / we**2
        B35 =  self.velocity * A33
        
        A55 = a33 * self.boy**3 / 12 + A33 * self.velocity**2 / we**2
        B55 = b33 * self.boy**3 / 12 + self.velocity**2 / we**2 * B33
        C55 = self.yogunluk * self.yer_cekimi * self.genislik * self.boy**3 / 12
        
        A53 = B33 * self.velocity / we**2
        B53 = - self.velocity * A33
        
        coef = np.array([[-(M + A33) * we**2 + C33, -B33 * we, -A35 * we**2, -B35 * we],      
                         [  B33 * we, -(M + A33) * we**2 + C33, B35 * we, -A35 * we**2],
                         [-A53 * we**2, -B53 * we, -(I + A55) * we**2 + C55, -B55 * we],
                         [  B53 * we, -A53 * we**2, B55 * we, -(I + A55) * we**2 + C55]])
        
        F = np.array([F3.real, F3.imag, F5.real, F5.imag])
        
        resu = np.linalg.solve(coef, F)
        z_real, z_imag = resu[0], resu[1]
        teta_real, teta_imag = resu[2], resu[3]
        
        heave = np.sqrt(z_real**2 + z_imag**2)
        pitch = np.sqrt(teta_real**2 + teta_imag**2)

        return heave, pitch, F, a33, b33
    

    def show_heave_pitch_rao(self, dalga_boyu):
        rao_heave = np.empty_like(dalga_boyu)
        rao_pitch = np.empty_like(dalga_boyu)

        for i in range(len(dalga_boyu)):
            lamda = dalga_boyu[i] * self.boy
            dalga_frekansi = np.sqrt(self.yer_cekimi * np.pi * 2 / lamda)
            dalga_katsayisi = dalga_frekansi**2 / self.yer_cekimi
            heave, pitch, F, a33, b33 = self.calculate_heave_pitch(dalga_frekansi, dalga_katsayisi)
            
            rao_heave[i] = heave / self.dalga_genligi
            rao_pitch[i] = pitch / self.dalga_genligi
            self.F3R.append(F[0])
            self.F3I.append(F[1])
            self.F5R.append(-F[2])
            self.F5I.append(F[3])
            self.a33.append(a33)
            self.b33.append(b33)
            
        plt.figure(figsize = (10, 4), dpi = 80)
        plt.grid()
        plt.plot(dalga_boyu, rao_heave)
        plt.title(r"$ROA_z$")
        plt.ylabel(r"$\frac{z}{A}$")
        plt.xlabel(r"$\frac{\lambda}{L}$")
        plt.savefig("heave_rao_v2")
        
        plt.figure(figsize = (10, 4), dpi = 80)
        plt.grid()
        plt.plot(dalga_boyu, rao_pitch)
        plt.title(r"$ROA_\theta$")
        plt.ylabel(r"$\frac{\theta}{A}$")
        plt.xlabel(r"$\frac{\lambda}{L}$")
        plt.savefig("pitch_rao_v2")
    

    def rms_periyod(self, dalga_boyu):
        lamda = dalga_boyu * self.boy
        dalga_frekansi = np.sqrt(self.yer_cekimi * np.pi * 2 / lamda)

        # GEMİNİN DENİZ DURUMU 4
        Hs = 1.88
        # REYLEİGN DAĞILIMI
        Sw = (8.1e-3 * self.yer_cekimi**2 / dalga_frekansi**5) * np.exp(-.032 * (self.yer_cekimi**2 / Hs**2) / dalga_frekansi**4)
        m0 = (quad( lambda w: Sw, 0, np.inf ))[0]
        
        # RMS DEĞERİ
        RMS = np.sqrt(np.mean( m0**2) )
        
        # ORTALAMA MERKEZ PERİYODU
        m1 = (quad( lambda dalga_frekansi: Sw * dalga_frekansi, 0, np.inf ))[0]
        T1 = 2 * np.pi * (m0 / m1)
        
        # ORTALAMA SIFIR GEÇME PERİYODU
        m2 = (quad( lambda dalga_frekansi: Sw * dalga_frekansi**2, 0, np.inf ))[0]
        Tz = 2 * np.pi * np.sqrt( m0 / m2 )
        
        # ORTALAMA TEPEDEN TEPEYE PERİYODU
        m4 = (quad( lambda dalga_frekansi: Sw * dalga_frekansi**4, 0, np.inf ))[0]
        Tc = 2 * np.pi * np.sqrt( m2 / m4 )

        return RMS, T1, Tz, Tc
    

    def save_table(self, dalga_boyu):
        df = pd.DataFrame([self.a33, self.b33, self.F3R, self.F3I, self.F5R, self.F5I], columns = np.round(dalga_boyu, 2),
                           index = ["a33", "a55", "F3R", "F3I", "F5R", "F5I"])
        
        df = df.round(2)
        dfi.export(df, "tablo2.png")

boy = 100 # GEMİNİN BOYU (LBP)
genislik = 20 # GEMİNİN TAM GENİŞLİĞİ
draft = 2.5 # GEMİNİN TAM YÜKLÜ DRAFTI
yogunluk = 1.025 # DENİZ SUYUNUN YOĞUNLUĞU
gemi = ShipDynamics(boy, genislik, draft, 1, 1, 135, .1, yogunluk, 9.808)

dalga_boyu = np.arange(.25, 6. + .25, step = .25)
gemi.show_heave_pitch_rao(dalga_boyu)
gemi.save_table(dalga_boyu)

dalga_boyu = 6

RMS, T1, Tz, Tc = gemi.rms_periyod(dalga_boyu)