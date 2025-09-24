#!/usr/bin/env python3
"""
Análisis Armónico del Sistema Biela-Manivela
Este script realiza el análisis completo del sistema biela-manivela según el README.md
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import os
import glob

# Configuración de matplotlib para gráficas en español
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (12, 8)

class BielaManivelaAnalysis:
    def __init__(self, data_file):
        """Inicializa el análisis con un archivo de datos"""
        self.data_file = data_file
        self.data = None
        self.frequency = self._extract_frequency_from_filename()
        self.load_data()

    def _extract_frequency_from_filename(self):
        """Extrae la frecuencia del nombre del archivo"""
        filename = os.path.basename(self.data_file)
        if "10Hz" in filename or "10hz" in filename:
            return 10
        elif "20Hz" in filename or "20hz" in filename:
            return 20
        elif "30Hz" in filename or "30hz" in filename:
            return 30
        return None

    def load_data(self):
        """Carga los datos del archivo"""
        if self.data_file.endswith('.txt'):
            # Para archivos .txt de Vernier Logger Pro
            try:
                self.data = pd.read_csv(self.data_file, sep='\t', skiprows=6)
                # Limpiar nombres de columnas
                self.data.columns = self.data.columns.str.strip()
                # Mapear nombres de columnas en español/inglés
                column_mapping = {
                    'Tiempo': 't',
                    't': 't',
                    'Posición': 'x',
                    'x': 'x',
                    'Velocidad 1': 'v',
                    'v 1': 'v',
                    'Aceleración 1': 'a',
                    'a 1': 'a',
                    'Angulo': 'theta',
                    'Velocidad 2': 'omega',
                    'v 2': 'omega',
                    'Aceleración 2': 'alpha',
                    'a 2': 'alpha'
                }
                self.data = self.data.rename(columns=column_mapping)

            except Exception as e:
                print(f"Error cargando archivo .txt: {e}")
                return None

        elif self.data_file.endswith('.csv'):
            # Para archivos .csv
            try:
                self.data = pd.read_csv(self.data_file)
                # Mapear nombres de columnas específicos para archivos .csv
                csv_column_mapping = {
                    'Último: Tiempo (s)': 't',
                    'Último: Angulo (rad)': 'theta',
                    'Último: Velocidad 1 (rad/s)': 'omega',
                    'Último: Aceleración 1 (rad/s²)': 'alpha',
                    'Último: Posición (m)': 'x',
                    'Último: Velocidad 2 (m/s)': 'v',
                    'Último: Aceleración 2 (m/s²)': 'a'
                }
                self.data = self.data.rename(columns=csv_column_mapping)

            except Exception as e:
                print(f"Error cargando archivo .csv: {e}")
                return None

        # Limpiar datos - eliminar filas con NaN
        self.data = self.data.dropna()
        print(f"Datos cargados: {len(self.data)} puntos")
        print(f"Columnas disponibles: {list(self.data.columns)}")

    def plot_angular_motion(self):
        """1. Análisis del Movimiento Angular"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        t = self.data['t']
        theta = self.data['theta']
        omega = self.data['omega']
        alpha = self.data['alpha']

        # Ángulo vs tiempo
        axes[0].plot(t, theta, 'b-', linewidth=2, label='Datos experimentales')
        axes[0].set_ylabel('Ángulo (rad)')
        axes[0].set_title(f'Movimiento Angular - {self.frequency} Hz')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Velocidad angular vs tiempo
        axes[1].plot(t, omega, 'r-', linewidth=2, label='Velocidad angular experimental')
        # Ajuste de velocidad constante si es MCU
        omega_mean = np.mean(omega[omega != 0])  # Excluir valores cero
        axes[1].axhline(y=omega_mean, color='k', linestyle='--',
                       label=f'ω promedio = {omega_mean:.2f} rad/s')
        axes[1].set_ylabel('Velocidad angular (rad/s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Aceleración angular vs tiempo
        axes[2].plot(t, alpha, 'g-', linewidth=2, label='Aceleración angular')
        axes[2].set_ylabel('Aceleración angular (rad/s²)')
        axes[2].set_xlabel('Tiempo (s)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(f'movimiento_angular_{self.frequency}Hz.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Evaluación del MCU
        omega_std = np.std(omega[omega != 0])
        print(f"\nEvaluación del MCU para {self.frequency} Hz:")
        print(f"Velocidad angular promedio: {omega_mean:.3f} rad/s")
        print(f"Desviación estándar: {omega_std:.3f} rad/s")
        print(f"¿Es MCU aproximadamente? {'Sí' if omega_std < 0.1 * abs(omega_mean) else 'No'}")

    def plot_linear_motion(self):
        """Análisis del Movimiento Lineal"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        t = self.data['t']
        x = self.data['x']
        v = self.data['v']
        a = self.data['a']

        # Posición vs tiempo
        axes[0].plot(t, x, 'b-', linewidth=2, label='Posición experimental')
        axes[0].set_ylabel('Posición (m)')
        axes[0].set_title(f'Movimiento Lineal del Pistón - {self.frequency} Hz')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Velocidad vs tiempo
        axes[1].plot(t, v, 'r-', linewidth=2, label='Velocidad experimental')
        axes[1].set_ylabel('Velocidad (m/s)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Aceleración vs tiempo
        axes[2].plot(t, a, 'g-', linewidth=2, label='Aceleración experimental')
        axes[2].set_ylabel('Aceleración (m/s²)')
        axes[2].set_xlabel('Tiempo (s)')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(f'movimiento_lineal_{self.frequency}Hz.png', dpi=300, bbox_inches='tight')
        plt.show()

    def fit_mas_equations(self):
        """Ajuste de ecuaciones del MAS"""
        t = self.data['t'].values
        x = self.data['x'].values
        v = self.data['v'].values
        a = self.data['a'].values

        # Definir funciones del MAS
        def mas_position(t, A0, A, omega, phi):
            return A0 + A * np.cos(omega * t + phi)

        def mas_velocity(t, A, omega, phi):
            return -omega * A * np.sin(omega * t + phi)

        def mas_acceleration(t, A, omega, phi):
            return -omega**2 * A * np.cos(omega * t + phi)

        try:
            # Ajuste de posición
            initial_guess_pos = [np.mean(x), (np.max(x) - np.min(x))/2, 2*np.pi*self.frequency, 0]
            popt_x, pcov_x = curve_fit(mas_position, t, x, p0=initial_guess_pos)

            # Ajuste de velocidad
            initial_guess_vel = [(np.max(v) - np.min(v))/2, 2*np.pi*self.frequency, 0]
            popt_v, pcov_v = curve_fit(mas_velocity, t, v, p0=initial_guess_vel)

            # Ajuste de aceleración
            initial_guess_acc = [(np.max(a) - np.min(a))/2, 2*np.pi*self.frequency, 0]
            popt_a, pcov_a = curve_fit(mas_acceleration, t, a, p0=initial_guess_acc)

            # Generar datos ajustados
            t_fit = np.linspace(t.min(), t.max(), 1000)
            x_fit = mas_position(t_fit, *popt_x)
            v_fit = mas_velocity(t_fit, *popt_v)
            a_fit = mas_acceleration(t_fit, *popt_a)

            # Graficar comparación
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            axes[0].plot(t, x, 'b-', alpha=0.7, label='Datos experimentales')
            axes[0].plot(t_fit, x_fit, 'r--', linewidth=2, label='Ajuste MAS')
            axes[0].set_ylabel('Posición (m)')
            axes[0].set_title(f'Ajuste MAS - {self.frequency} Hz')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

            axes[1].plot(t, v, 'b-', alpha=0.7, label='Datos experimentales')
            axes[1].plot(t_fit, v_fit, 'r--', linewidth=2, label='Ajuste MAS')
            axes[1].set_ylabel('Velocidad (m/s)')
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()

            axes[2].plot(t, a, 'b-', alpha=0.7, label='Datos experimentales')
            axes[2].plot(t_fit, a_fit, 'r--', linewidth=2, label='Ajuste MAS')
            axes[2].set_ylabel('Aceleración (m/s²)')
            axes[2].set_xlabel('Tiempo (s)')
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()

            plt.tight_layout()
            plt.savefig(f'ajuste_mas_{self.frequency}Hz.png', dpi=300, bbox_inches='tight')
            plt.show()

            print(f"\nParámetros del ajuste MAS para {self.frequency} Hz:")
            print(f"Posición: A₀={popt_x[0]:.4f}m, A={popt_x[1]:.4f}m, ω={popt_x[2]:.3f}rad/s, φ={popt_x[3]:.3f}rad")
            print(f"Velocidad: A={popt_v[0]:.4f}m/s, ω={popt_v[1]:.3f}rad/s, φ={popt_v[2]:.3f}rad")
            print(f"Aceleración: A={popt_a[0]:.4f}m/s², ω={popt_a[1]:.3f}rad/s, φ={popt_a[2]:.3f}rad")

            return popt_x, popt_v, popt_a

        except Exception as e:
            print(f"Error en ajuste MAS: {e}")
            return None, None, None

    def plot_position_vs_angle(self):
        """2. Comparación entre Espacio Angular y Lineal"""
        theta = self.data['theta'].values
        x = self.data['x'].values

        plt.figure(figsize=(10, 8))
        plt.plot(theta, x, 'b-', linewidth=2, alpha=0.7, label='x(θ) experimental')

        # Intentar ajuste sinusoidal
        def sinusoidal(theta, A0, A, phi):
            return A0 + A * np.cos(theta + phi)

        try:
            # Filtrar datos válidos
            valid_mask = ~np.isnan(theta) & ~np.isnan(x)
            theta_clean = theta[valid_mask]
            x_clean = x[valid_mask]

            if len(theta_clean) > 3:
                initial_guess = [np.mean(x_clean), (np.max(x_clean) - np.min(x_clean))/2, 0]
                popt, pcov = curve_fit(sinusoidal, theta_clean, x_clean, p0=initial_guess)

                theta_fit = np.linspace(theta_clean.min(), theta_clean.max(), 1000)
                x_fit = sinusoidal(theta_fit, *popt)

                plt.plot(theta_fit, x_fit, 'r--', linewidth=2,
                        label=f'Ajuste: x = {popt[0]:.3f} + {popt[1]:.3f}cos(θ + {popt[2]:.3f})')

                # Calcular R²
                x_pred = sinusoidal(theta_clean, *popt)
                ss_res = np.sum((x_clean - x_pred) ** 2)
                ss_tot = np.sum((x_clean - np.mean(x_clean)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                print(f"\nAjuste x(θ) para {self.frequency} Hz:")
                print(f"Parámetros: A₀={popt[0]:.4f}m, A={popt[1]:.4f}m, φ={popt[2]:.3f}rad")
                print(f"R² = {r_squared:.4f}")
                print(f"¿Es sinusoidal? {'Sí' if r_squared > 0.9 else 'No - requiere análisis por partes'}")

        except Exception as e:
            print(f"Error en ajuste sinusoidal: {e}")

        plt.xlabel('Ángulo θ (rad)')
        plt.ylabel('Posición x (m)')
        plt.title(f'Posición del Pistón vs Ángulo de la Manivela - {self.frequency} Hz')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(f'posicion_vs_angulo_{self.frequency}Hz.png', dpi=300, bbox_inches='tight')
        plt.show()

    def spectral_analysis(self):
        """3. Análisis Espectral de la Aceleración (DFT)"""
        t = self.data['t'].values
        a = self.data['a'].values

        # Seleccionar intervalo periódico (eliminar transitorios)
        start_idx = len(a) // 10  # Eliminar primer 10%
        end_idx = -len(a) // 10   # Eliminar último 10%

        t_periodic = t[start_idx:end_idx]
        a_periodic = a[start_idx:end_idx]

        # Parámetros para FFT
        dt = np.mean(np.diff(t_periodic))
        fs = 1/dt  # Frecuencia de muestreo
        N = len(a_periodic)

        # Aplicar DFT
        A_fft = fft(a_periodic)
        freqs = fftfreq(N, dt)

        # Solo frecuencias positivas
        positive_freqs = freqs[:N//2]
        amplitudes = 2.0/N * np.abs(A_fft[:N//2])
        phases = np.angle(A_fft[:N//2])

        # Encontrar picos principales
        peak_indices = []
        threshold = 0.1 * np.max(amplitudes)

        for i in range(1, len(amplitudes)-1):
            if amplitudes[i] > threshold and amplitudes[i] > amplitudes[i-1] and amplitudes[i] > amplitudes[i+1]:
                peak_indices.append(i)

        # Ordenar por amplitud y tomar los dos primeros
        peak_indices = sorted(peak_indices, key=lambda i: amplitudes[i], reverse=True)[:2]

        if len(peak_indices) >= 2:
            f1, f2 = positive_freqs[peak_indices[0]], positive_freqs[peak_indices[1]]
            A1, A2 = amplitudes[peak_indices[0]], amplitudes[peak_indices[1]]
            phi1, phi2 = phases[peak_indices[0]], phases[peak_indices[1]]

            # Reconstruir señal
            a_reconstructed = A1 * np.cos(2*np.pi*f1*t_periodic + phi1) + A2 * np.cos(2*np.pi*f2*t_periodic + phi2)

            # Gráficas
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Espectro de amplitudes
            axes[0,0].stem(positive_freqs, amplitudes, basefmt=' ')
            axes[0,0].set_xlabel('Frecuencia (Hz)')
            axes[0,0].set_ylabel('Amplitud')
            axes[0,0].set_title('Espectro de Amplitudes - DFT')
            axes[0,0].grid(True, alpha=0.3)
            axes[0,0].set_xlim(0, 10*self.frequency)  # Limitar vista

            # Marcar picos principales
            for i, peak_idx in enumerate(peak_indices):
                axes[0,0].plot(positive_freqs[peak_idx], amplitudes[peak_idx], 'ro', markersize=8,
                              label=f'f{i+1}={positive_freqs[peak_idx]:.2f}Hz')
            axes[0,0].legend()

            # Señal original vs tiempo
            axes[0,1].plot(t_periodic, a_periodic, 'b-', alpha=0.7, label='Aceleración original')
            axes[0,1].plot(t_periodic, a_reconstructed, 'r--', linewidth=2, label='Reconstruida (2 armónicos)')
            axes[0,1].set_xlabel('Tiempo (s)')
            axes[0,1].set_ylabel('Aceleración (m/s²)')
            axes[0,1].set_title('Comparación: Original vs Reconstruida')
            axes[0,1].grid(True, alpha=0.3)
            axes[0,1].legend()

            # Espectro de fases
            axes[1,0].stem(positive_freqs, phases, basefmt=' ')
            axes[1,0].set_xlabel('Frecuencia (Hz)')
            axes[1,0].set_ylabel('Fase (rad)')
            axes[1,0].set_title('Espectro de Fases')
            axes[1,0].grid(True, alpha=0.3)
            axes[1,0].set_xlim(0, 10*self.frequency)

            # Error de reconstrucción
            error = a_periodic - a_reconstructed
            axes[1,1].plot(t_periodic, error, 'g-', linewidth=1)
            axes[1,1].set_xlabel('Tiempo (s)')
            axes[1,1].set_ylabel('Error (m/s²)')
            axes[1,1].set_title('Error de Reconstrucción')
            axes[1,1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'analisis_espectral_{self.frequency}Hz.png', dpi=300, bbox_inches='tight')
            plt.show()

            # Resultados numéricos
            mse = np.mean(error**2)
            print(f"\nAnálisis Espectral para {self.frequency} Hz:")
            print(f"Primer armónico:  f₁ = {f1:.3f} Hz, A₁ = {A1:.4f} m/s², φ₁ = {phi1:.3f} rad")
            print(f"Segundo armónico: f₂ = {f2:.3f} Hz, A₂ = {A2:.4f} m/s², φ₂ = {phi2:.3f} rad")
            print(f"Error cuadrático medio: {mse:.6f}")
            print(f"Relación f₂/f₁ = {f2/f1:.2f}")

            return f1, f2, A1, A2, phi1, phi2

        else:
            print("No se encontraron suficientes picos en el espectro")
            return None

    def physical_interpretation(self):
        """4. Interpretación Física de los Armónicos"""
        print(f"\n=== INTERPRETACIÓN FÍSICA - {self.frequency} Hz ===")
        print("\n1. ORIGEN DEL PRIMER ARMÓNICO:")
        print("   - Corresponde a la componente principal del MAS")
        print("   - Relacionado con el término -rω²cos(θ) en la ecuación de aceleración")
        print("   - Representa el movimiento fundamental del pistón")

        print("\n2. ORIGEN DEL SEGUNDO ARMÓNICO:")
        print("   - Aparece por la longitud finita de la biela (l)")
        print("   - Corresponde al término (r²ω²/l)cos(2θ) en la ecuación teórica")
        print("   - Corrección geométrica que aleja el sistema del MAS ideal")

        print("\n3. RELACIÓN CON MOTORES DE COMBUSTIÓN:")
        print("   - En motores de 4 cilindros, el segundo armónico puede causar vibraciones")
        print("   - La frecuencia del segundo armónico (2f₁) coincide con la frecuencia de explosión")
        print("   - Importante para el diseño de contrapesos y amortiguadores")

        print("\n4. IMPLICACIONES EN VIBRACIONES:")
        print("   - El segundo armónico contribuye a vibraciones del motor")
        print("   - Su amplitud depende de la relación r/l (radio manivela / longitud biela)")
        print("   - Motores modernos minimizan esta relación para reducir vibraciones")

    def complete_analysis(self):
        """Ejecuta el análisis completo"""
        print(f"=== ANÁLISIS COMPLETO DEL SISTEMA BIELA-MANIVELA ===")
        print(f"Archivo: {self.data_file}")
        print(f"Frecuencia: {self.frequency} Hz")
        print(f"Puntos de datos: {len(self.data)}")

        # 1. Análisis del movimiento
        print("\n1. ANÁLISIS DEL MOVIMIENTO ANGULAR...")
        self.plot_angular_motion()

        print("\n2. ANÁLISIS DEL MOVIMIENTO LINEAL...")
        self.plot_linear_motion()

        print("\n3. AJUSTE DE ECUACIONES MAS...")
        self.fit_mas_equations()

        # 2. Comparación angular vs lineal
        print("\n4. ANÁLISIS POSICIÓN vs ÁNGULO...")
        self.plot_position_vs_angle()

        # 3. Análisis espectral
        print("\n5. ANÁLISIS ESPECTRAL...")
        spectral_results = self.spectral_analysis()

        # 4. Interpretación física
        self.physical_interpretation()

        return spectral_results

def analyze_all_files():
    """Analiza todos los archivos de datos disponibles"""
    # Buscar archivos de datos
    txt_files = glob.glob("biela-manivela-*Hz*.txt")
    csv_files = glob.glob("*hz*.csv")

    all_files = txt_files + csv_files
    all_files = sorted(all_files)

    print(f"Archivos encontrados: {len(all_files)}")
    for file in all_files:
        print(f"  - {file}")

    results = {}

    for data_file in all_files:
        try:
            print(f"\n{'='*60}")
            print(f"ANALIZANDO: {data_file}")
            print(f"{'='*60}")

            analyzer = BielaManivelaAnalysis(data_file)
            if analyzer.data is not None:
                result = analyzer.complete_analysis()
                results[data_file] = result
            else:
                print(f"Error cargando {data_file}")

        except Exception as e:
            print(f"Error procesando {data_file}: {e}")
            continue

    return results

def generate_summary_report(results):
    """Genera un reporte resumen de todos los análisis"""
    print(f"\n{'='*80}")
    print("REPORTE RESUMEN - ANÁLISIS ARMÓNICO SISTEMA BIELA-MANIVELA")
    print(f"{'='*80}")

    # Crear tabla resumen
    summary_data = []

    for file, result in results.items():
        if result is not None:
            frequency = None
            if "10Hz" in file or "10hz" in file:
                frequency = 10
            elif "20Hz" in file or "20hz" in file:
                frequency = 20
            elif "30Hz" in file or "30hz" in file:
                frequency = 30

            f1, f2, A1, A2, phi1, phi2 = result
            summary_data.append({
                'Archivo': file,
                'Freq_Motor': frequency,
                'f1_Hz': f1,
                'f2_Hz': f2,
                'A1': A1,
                'A2': A2,
                'Ratio_f2_f1': f2/f1 if f1 > 0 else 0,
                'Ratio_A2_A1': A2/A1 if A1 > 0 else 0
            })

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        print("\nTABLA RESUMEN DE ARMÓNICOS:")
        print(df_summary.to_string(index=False, float_format='%.3f'))

        # Guardar tabla
        df_summary.to_csv('resumen_armonicos.csv', index=False, float_format='%.4f')
        print("\nTabla guardada como: resumen_armonicos.csv")

        # Análisis estadístico
        print(f"\nANÁLISIS ESTADÍSTICO:")
        print(f"Relación f₂/f₁ promedio: {df_summary['Ratio_f2_f1'].mean():.3f} ± {df_summary['Ratio_f2_f1'].std():.3f}")
        print(f"Relación A₂/A₁ promedio: {df_summary['Ratio_A2_A1'].mean():.3f} ± {df_summary['Ratio_A2_A1'].std():.3f}")

        # Validación teórica
        print(f"\nVALIDACIÓN TEÓRICA:")
        ratio_f_theoretical = 2.0
        ratio_f_experimental = df_summary['Ratio_f2_f1'].mean()
        error_f = abs(ratio_f_experimental - ratio_f_theoretical) / ratio_f_theoretical * 100

        print(f"f₂/f₁ teórico: {ratio_f_theoretical:.1f}")
        print(f"f₂/f₁ experimental: {ratio_f_experimental:.3f}")
        print(f"Error relativo: {error_f:.1f}%")
        print(f"¿Coincide con teoría? {'Sí' if error_f < 10 else 'No'}")

if __name__ == "__main__":
    # Ejecutar análisis completo
    results = analyze_all_files()
    generate_summary_report(results)

    print(f"\n{'='*60}")
    print("ANÁLISIS COMPLETADO")
    print("Archivos generados:")
    print("  - Gráficas: movimiento_angular_*Hz.png")
    print("  - Gráficas: movimiento_lineal_*Hz.png")
    print("  - Gráficas: ajuste_mas_*Hz.png")
    print("  - Gráficas: posicion_vs_angulo_*Hz.png")
    print("  - Gráficas: analisis_espectral_*Hz.png")
    print("  - Tabla: resumen_armonicos.csv")
    print(f"{'='*60}")