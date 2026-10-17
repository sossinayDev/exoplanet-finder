CACHE_FILE_PATH = "planetlibcache.json"

import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("QtAgg")
import lightkurve as lk
from datetime import datetime
from colorama import Fore
from time import time
import astropy.units as u
from rich.progress import Progress # Q: https://www.geeksforgeeks.org/python/progress-bars-in-python/
from astroquery.mast import Catalogs
import json
import batman
from scipy.optimize import curve_fit
print(Fore.YELLOW + "planetlib: Libraries loaded" + Fore.RESET)

SOLAR_RADIUS = 6.960 * 10**8 * u.m # Wert aus Formelsammlung (8te Auflage, p.211)
JUPITER_RADIUS = 71.492 * 10**6 * u.m # Ebenfalls von dort (p.212)
G = 6.674 * 10**-11 * u.m**3 / (u.kg * u.s**2) # Gravitationskonstante, Formelsammlung, letzte Seite

VERSION = "v0.52"

def clean(v):
        if v is None or v is np.ma.masked:
            return None
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and np.isnan(v):
            return None
        return v or None

def pct_to_mmag(depth_pct):
    flux_ratio = 1 - depth_pct/100

    delta_mag = -2.5 * np.log10(flux_ratio) * u.mag
    return delta_mag.to(u.mmag)

def strfunix(unix,string:str="%d/%m/%Y - %H:%M:%S"):
    return datetime.fromtimestamp(unix).strftime(string)

def remove_outliers(l, n=1):
    key = lambda x: x.value if isinstance(x, u.Quantity) else x
    s = sorted(l, key=key)
    return s[n:len(s) - n]
    
def ao(num,arr,pos,inv:bool=False,useindex=None):
    sum = 0
    i = 0
    try:
        for j in range(num):
            if inv:
                if isinstance(useindex, int): sum += arr[pos-j][useindex]
                else: sum += arr[pos-j]
                i += 1
            else:
                if isinstance(useindex, int): sum += arr[pos+j][useindex]
                else: sum += arr[pos+j]
                i += 1
    except:
        pass
    if i == 0: return 0
    return sum/i

# Custom exceptions (Errors) Source: https://www.geeksforgeeks.org/python/define-custom-exceptions-in-python/
class NoDataFound(Exception):
    # Called when parsing satellite data was not found (! NOT WHEN INDEX TO HIGH !)
    def __init__(self, message = "No data was found"):
        self.message = message
        super().__init__(message)
    
    def __str__(self):
        return f"{self.message}"

class NotAnalyzed(Exception):
    # Called when a batman fit is called on transit that hasn't been analyzed yet
    def __init__(self, message = "Transit is not yet analyzed"):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"

class FittingFailed(Exception):
    # Called when a batman fit fails
    def __init__(self, message = "Transit fitting failed"):
        self.message = message
        super().__init__(message)

    def __str__(self):
        return f"{self.message}"

class transit:
    times = []
    values = []
    errors = []
    combined = {}
    is_parsed = False

    start = 0
    end = 0
    depth = 0
    noise_range = []
    period = 0
    inclination = 0
    rp_rs = 0
    semi_major = 0
    limb_darkening = [0, 0]

    deep_analyzed = False
    analyzed = False

    duration_s = 0

    def __init__(self,start,end,times,values,errors,noiserange,period):
        self.noise_range = noiserange

        self.times = times
        self.values = values
        self.errors = errors

        self.period = period

        self.start = start
        self.end = end

        self.start_index = self.times.index(self.start)
        self.end_index = self.times.index(self.end)

        self.is_parsed = False

        self.analyze()

    def parse(self):
        self.is_parsed = False
        self.combined = {}
        i = 0
        for tstamp in self.times:
            self.combined[str(tstamp)] = {
                "v": self.values[i],
                "e": self.errors[i]
            }
            i += 1
        self.is_parsed = True

  
    def analyze(self, deep_analyze:bool=False):
        if not self.is_parsed: self.parse()
        data_range = sorted(list(self.combined.items())[self.start_index:self.end_index], key=lambda item: item[1]["v"]) # https://www.geeksforgeeks.org/python/python-sort-a-dictionary/
        lowest_values = data_range[:int(len(data_range)*15/100)]
        average_minimum_value = data_range[0][1]["v"]
        error = data_range[0][1]["e"]
        if len(lowest_values) > 2:
            average_minimum_value = 0
            error = 0
            i = 0
            for pair in lowest_values:
                val_pair = pair[1]
                average_minimum_value += val_pair["v"] # Average of the lowest 15% of vals
                error += val_pair["e"] ** 2
                i += 1
            average_minimum_value /= i
            error = math.sqrt(error) / i # Fehler geht mit der Wurzel, siehe Formelbuch S. 64
            
        median = (self.noise_range[0]+self.noise_range[1])/2
        self.depth = (median-average_minimum_value)/median

        self.duration_s = self.end - self.start
        self.analyzed = True

        if deep_analyze:
            self.deep_analyze()
            if self.deep_analyzed:
                pass
                # self.log_dbg(f"Deep analysis successful: Rp/Rs={self.rp_rs:.4f}, a={self.semi_major:.4f}, inc={self.inclination:.2f}°")
            else:
                self.log_err("Deep analysis failed")


    def deep_analyze(self):
        # self.log_dbg("Starting deep analysis of transit...")
        self.deep_analyzed = False
        if not self.analyzed:
            self.log_err("Transit is not yet analyzed, cannot perform deep analysis.")
            raise NotAnalyzed

        try:
            fitting_data = self.fit_transit()
            self.fit_transit_params = fitting_data
            self.depth = fitting_data[1]**2
            per = self.period.to(u.day).value if hasattr(self.period, "unit") else float(self.period)
            self.duration_s = per * 86400 / (math.pi * fitting_data[2])
            self.rp_rs = fitting_data[1]        # Verhältnis von Stern- und Planet-radius
            self.semi_major = fitting_data[2]
            self.inclination = fitting_data[3]
    
            self.deep_analyzed = True
        except Exception as e:
            self.log_err(f"Deep analysis failed: {e}")

       

    def fit_transit(self, ld=(0.35, 0.25)):
        if self.period == 0: raise ValueError("Period was not passed correctly")
        try:

            # https://lkreidberg.github.io/batman/docs/html/
            # https://lkreidberg.github.io/batman/docs/html/tutorial.html?highlight=fitting

            # Zeiten in Tage umwandeln (Unix ist in sekunden, batman braucht tage)
            duration_days = self.duration_s / 86400
            times = np.array(self.times) / 86400
            t0_guess = (self.start + self.end) / 2 / 86400
            # self.log_dbg("Times converted")

            # Daten croppen, links bzw rechts je 2*die dauer des transits (dauer aus basic messungen)
            m = np.abs(times - t0_guess) < 2 * duration_days
            t = times[m] - t0_guess 
            f = np.array(self.values)[m]
            e = np.array(self.errors)[m]
            # self.log_dbg(f"Data cropped")

            # periode in float converten
            per = self.period.to(u.day).value if hasattr(self.period, "unit") else float(self.period)

            # batmna initialisieren
            p = batman.TransitParams()
            p.t0, p.per = 0.0, per                     # periode aus periodogramm
            p.rp = math.sqrt(max(self.depth, 1e-6))    # RP ist das verhältnis von Stern- und Planet-radius
            p.a = per / (math.pi * duration_days)      # grobe schätzung der grossen Halbachse
            p.inc, p.ecc, p.w = 88.0, 0.0, 90.0        # Arbitäre Zahlen, werden durch modell berechnet
            p.u, p.limb_dark = list(ld), "quadratic"
            mdl = batman.TransitModel(p, t)
            # self.log_dbg("Transit model initialized")

            def model(_, t0, rp, a, inc):
                p.t0, p.rp, p.a, p.inc = t0, rp, a, inc
                return mdl.light_curve(p)

            # Anpassen des Modelles auf die Messdaten
            p0 = [0.0, p.rp, p.a, p.inc]
            bounds = ([-duration_days, 0.001, 1.5, 60], [duration_days, 0.5, 200, 90])
            p0 = list(np.clip(p0, np.array(bounds[0]) + 1e-6, np.array(bounds[1]) - 1e-6))
            popt, pcov = curve_fit(model, t, f, p0=p0, sigma=e,
                                absolute_sigma=True, bounds=bounds)
            # self.log_dbg("Transit fitting completed")
            self.fit_params = dict(zip(["t0", "rp", "a", "inc"], popt))
            self.fit_errors = dict(zip(["t0", "rp", "a", "inc"], np.sqrt(np.diag(pcov))))
            self.rp_rs = popt[1]
            # self.log_dbg("Saved fitting parameters")
            self.limb_darkening = ld
            return popt
        except Exception as e:
            raise FittingFailed(f"Transit fitting failed: {e}")

    def model_curve(self, n=500):
        if not self.deep_analyzed: raise NotAnalyzed
        dur_d = (self.end - self.start) / 86400
        center = (self.start + self.end) / 2 / 86400
        t = np.linspace(-2 * dur_d, 2 * dur_d, n)      # same window as the fit

        per = self.period.to(u.day).value if hasattr(self.period, "unit") else float(self.period)
        fp = self.fit_params
        p = batman.TransitParams()
        p.t0, p.per, p.rp, p.a, p.inc = fp["t0"], per, fp["rp"], fp["a"], fp["inc"]
        p.ecc, p.w = 0.0, 90.0
        p.u, p.limb_dark = list(self.limb_darkening), "quadratic"
        flux = batman.TransitModel(p, t).light_curve(p)
        return (t + center) * 86400, flux               # back to unix seconds
    
    def save(self):
        return {
            "start": self.start,
            "start_index": self.start_index,
            "end": self.end,
            "end_index": self.end_index,
            "depth": self.depth,
            "duration_s": self.duration_s,
            "noise_range": self.noise_range,
            "period": self.period,
            "inclination": self.inclination,
            "rp_rs": self.rp_rs,
            "semi_major": self.semi_major,
            "analyzed": self.analyzed,
            "deep_analyzed": self.deep_analyzed
        }
        

    def showshort(self):
        t = f"START: {strfunix(self.start)} | END: {strfunix(self.end)}"
        if self.analyzed: t += f" | Depth: {self.depth*100:.2f}% | Duration: {self.duration_s:.2f}s"
        if self.deep_analyzed: t += f" | Rp/Rs: {self.rp_rs:.4f} | a: {self.semi_major:.4f} | inc: {self.inclination:.2f}°"
        return t

    def log_err(self, message):
        print(Fore.RED + f"[Transit {strfunix(self.start)} - {strfunix(self.end)}] ERROR: {message}" + Fore.RESET)

    def log_wrn(self, message):
        print(Fore.YELLOW + f"[Transit {strfunix(self.start)} - {strfunix(self.end)}] WARNING: {message}" + Fore.RESET)

    def log_dbg(self, message):
        print(f"[Transit {strfunix(self.start)} - {strfunix(self.end)}] DEBUG: {message}" + Fore.RESET)

# Measurements that can be used to easily access curves etc.
class measurement:
    target = ""
    index = 0
    pixelfile = None
    curve = None
    filled_curve = None
    period = None
    is_loaded = False
    is_parsed = False
    search = None
    noise_range = None
    data_source = None

    data_values = []
    data_times = []
    data_errors = []
    data_pairs = {}

    mask_array = None
    data_masked = False

    transits = None

    stopwatch_start = 0
    debug_times = {}

    is_analyzed = False

    discontinuities = None

    def __init__(self, star_name, measurement_index, search, autoload: bool = False):
        self.data_values = []
        self.data_times = []
        self.data_errors = []
        self.data_pairs = {}
        self.target = star_name
        self.search = search
        self.index = measurement_index
        if autoload:
            self.load_data_from_sat(autoparse=True)
            self.get_data_source()

    
    def load_data_from_sat(self,autoparse=False):
        if len(self.search) < self.index-1:
            # The observation depth is too high and not enough data is provided by the satellites.
            raise IndexError("Index for search out of range, maximum depth reached.")
        
        # Data exists, download it
        self.start_stopwatch()
        self.pixelfile = self.search[self.index].download()
        self.stop_stopwatch("Download duration")
        self.start_stopwatch()
        # Download and smoothen the curve
        self.curve = self.pixelfile.to_lightcurve().remove_outliers(sigma=6)
        # self.curve.plot()
        # plt.show()
        # Convert the curve to a periodogram for period calculation for folding
        self.periodogram = self.curve.to_periodogram("bls")
        self.period = self.periodogram.period_at_max_power
        self.curve.fold(self.period).remove_nans();
        self.filled_curve = self.curve.flatten()
        self.stop_stopwatch("Enhancing duration")
        
        self.is_loaded = True
        if autoparse: self.parse()

    def get_data_source(self):
        if self.pixelfile == None:
            return
        self.data_source = self.pixelfile.mission
        return self.data_source

    def load(self,ld):
        self.period = ld.get("self.period", 0) * u.day
        self.is_loaded = ld.get("is_loaded")
        self.is_parsed = ld.get("is_parsed")
        self.is_analyzed = ld.get("is_analyzed")
        self.noise_range = tuple(ld.get("noise_range", (None, None)))
        self.data_values = ld.get("data_values")
        self.data_errors = ld.get("data_errors")
        self.data_times = ld.get("data_times")
        self.mask_array = ld.get("mask_array")
        self.data_masked = ld.get("data_masked")
        self.debug_times = ld.get("debug_times")
        self.stopwatch_start = ld.get("stopwatch_start")
        self.discontinuities = ld.get("discontinuities")
        self.transits = []

        for t in ld["transits"]:
            new_inst = transit(t["start"],t["end"],self.data_times,self.data_values,self.noise_range,self.period)
            new_inst.depth = t["depth"]
            new_inst.duration_s = t["duration_s"]
            new_inst.noise_range = t["noise_range"]
            new_inst.semi_major = t["semi_major"]
            new_inst.rp_rs = t["rp_rs"]
            new_inst.inclination = t["inclination"]
            new_inst.analyzed = t["analyzed"]
            self.transits.append(new_inst)

    def parse(self):
        self.start_stopwatch()
        temp_time_list = list(self.filled_curve.time)
        temp_value_list = list(self.filled_curve.flux)
        temp_err_list = list(self.filled_curve.flux_err)

        self.data_values = []
        self.data_times = []
        self.data_errors = []
        self.data_pairs = {}


        # Iterate through Quantity objects and turn them into something I understand
        for scanning_index in range(len(temp_time_list)):
            t = temp_time_list[scanning_index].to_value("unix")
            if t < 10:
                continue
            v = float(temp_value_list[scanning_index].value)
            e = float(temp_err_list[scanning_index].value)

            self.data_times.append(t)
            self.data_values.append(v)
            self.data_errors.append(e)
            self.data_pairs[t] = [v,e]
        
        self.is_parsed = True # Save progress
        self.stop_stopwatch("Parsing time")

    # Analysis
    def analyze(self,mask:bool=False):
        self.get_noise_range()
        self.find_discontinuities()
        self.find_transits()
        self.is_analyzed = True
        if mask:
            self.mask_data()
            self.is_analyzed = False
            self.transits = []
            self.find_transits()
            self.is_analyzed = True

    def mask_data(self):
        self.mask_array = self.curve.create_transit_mask(period=self.period, duration=(self.transits[0].end-self.transits[0].start)/60/60/24/2, transit_time=self.transits[0].start)
        self.filled_curve = self.curve.flatten(mask=self.mask_array)
        self.data_masked = True

    def get_avg_scan_interval(self):
        if len(self.data_times) == 0: return 0.01
        return (self.end() - self.start()) / len(self.data_times)

    
    def get_noise_range(self):
        def windowed_range(values):
            minsum = 0
            maxsum = 0
            counter = 0
            for i in range(math.floor(len(values)/15)):
                l_min = 1
                l_max = 1
                for j in range(15):
                    try:
                        v = values[i+j]
                        if v > l_max: l_max = v
                        if v < l_min: l_min = v
                    except IndexError:
                        break
                minsum += l_min
                maxsum += l_max
                counter += 1
            if counter == 0:
                return (min(values), max(values))
            return (minsum/counter, maxsum/counter)

        rough_min, rough_max = windowed_range(self.data_values)

        margin = (rough_max - rough_min) * 0.5
        clean_values = [v for v in self.data_values if v > rough_min - margin]

        if len(clean_values) < 15:
            clean_values = self.data_values

        n_min, n_max = windowed_range(clean_values)
        self.noise_range = (n_min, n_max)
        return (n_min, n_max)
    

    def find_discontinuities(self): # Find gaps between values
        disc_list = []
        avg_interval_seconds = self.get_avg_scan_interval()

        last_time = self.data_times[0]
        for scan_time in self.data_times[1:]:
            if scan_time-last_time > avg_interval_seconds * 10:
                disc_list.append({
                    "from": float(last_time),
                    "to": float(scan_time)
                }.copy())
            last_time = scan_time

        self.discontinuities = disc_list
        return disc_list

    def check_for_discontinuity(self,fromto_indexes) -> bool:
        start_index, end_index = fromto_indexes
        if end_index > len(self.data_times)-1: return True
        start_time, end_time = (self.data_times[start_index],self.data_times[end_index])
        for disc in self.discontinuities:
            f = disc["from"]
            t = disc["to"]
            if start_time > f and start_time < t: return True
            if end_time > f and end_time < t: return True
            if f > start_time and f < end_time: return True
            if t > start_time and t < end_time: return True
        return False


    def get_out_of_noise_points(self):
        n_min, n_max = self.get_noise_range()
        res = []
        for i in range(len(self.data_times)):
            value = self.data_values[i]
            if value > n_max or value < n_min: res.append(i)
        return res
    
    def find_transits(self):

        flagged_points = self.get_out_of_noise_points()
        n_min, n_max = self.noise_range

        # epsilon = 1+(n_max-n_min)*0.1
        epsilon = 1

        transits = []
        added_until = 0

        

        for point_index in flagged_points:
            if point_index < added_until: continue
            current_value = self.data_values[point_index]
            if current_value < 1:
                is_strong_negative = True
                valid = True
                for j in range(-2,3):
                    if point_index+j > len(self.data_values)-1:
                        valid = False
                        break
                    if self.data_values[point_index+j] > n_min:
                        is_strong_negative = False
                        break
                if not is_strong_negative: continue
                v = current_value
                ci = point_index
                while v < n_max and ao(5,self.data_values,ci,inv=True,useindex=0)<epsilon and valid:
                    ci -= 1
                    if ci < 1:
                        valid = False
                    v = self.data_values[ci]
                transit_start = ci
                v = current_value
                ci = point_index
                while v < n_max and ao(5,self.data_values,ci,useindex=0)<epsilon and valid:
                    ci += 1
                    if ci > len(self.data_values)-1:
                        valid = False
                        break
                    v = self.data_values[ci]
                transit_end = ci

                if self.check_for_discontinuity((transit_start,transit_end)): valid = False

                if valid:
                    transits.append(transit(self.data_times[transit_start],self.data_times[transit_end],self.data_times,self.data_values,self.data_errors,self.noise_range,self.period))
                    added_until = transit_end
        
        self.transits = transits



    
    # Export/Loading
    def save(self):
        sd = {
            "target": self.target,
            "index": self.index,
            "self.period": float(str(self.period).split(" ")[0]),
            "is_loaded": self.is_loaded,
            "is_parsed": self.is_parsed,
            "is_analyzed": self.is_analyzed,
            "noise_range": self.noise_range,
            "data_values": self.data_values,
            "data_errors": self.data_errors,
            "data_times": self.data_times,
            "mask_array": self.mask_array,
            "data_masked": self.data_masked,
            "debug_times": {},
            "stopwatch_start": self.stopwatch_start,
            "discontinuities": self.discontinuities,
            "transits": []
        }
        for t in self.transits:
            sd["transits"].append(t.save())
        return sd

    def export_as_csv(self, path: str):
        self.start_stopwatch()

        csv_string = "Time (unix timestamp),Relative flux,Error"

        data = self.data_pairs
        for key in data.keys():
            value = data[key][0]
            error = data[key][1]
            csv_string+=f"\n{key},{value},{error}"

        with open(path,"w",encoding="utf-8") as file:
            file.write(csv_string)
        self.stop_stopwatch("CSV export")
    
    def showshort(self):
        return f"""
Measurement {self.target}#{self.index}
|  target: {self.target}
|  index: {self.index}
|  period: {self.period}
|  is_loaded: {self.is_loaded}
|  is_parsed: {self.is_parsed}
|  is_analyzed: {self.is_analyzed}
|  noise_range: {self.noise_range}
|  transits: {self.transits}
|  discontinuities: {self.discontinuities}
|  data_points: {len(self.data_values)}
"""

    
    # Plotting      Source: https://matplotlib.org/stable/plot_types/basic/scatter_plot.html
    def show_plot(self,**kwargs):
        if kwargs.get("smooth",False):
            s_times = []
            s_vals = []
            i = 0
            while i < len(self.data_times)-6:
                s_times.append(self.data_times[i])
                s_vals.append(ao(5,self.data_values,i))
                i+=5
            s_times.append(self.data_times[i])
            s_vals.append(ao(len(self.data_times)-1-i,self.data_times,i))
            # print(s_times)
            # print(s_vals)
            plt.plot(s_times, s_vals)
        else:
            if kwargs.get("dots",True):
                plt.scatter(self.data_times,self.data_values)
            else:
                plt.plot(self.data_times,self.data_values)
        plt.xlabel("Time (unix timestamp)")
        plt.ylabel("Relative flux")
        plt.xlim(self.start(),self.end())
        plt.ylim(0,2)
        plt.title(f"Measurement #{self.index} ({self.target})")

        if kwargs.get("annotated", False):
            error_mode = kwargs.get("errors", None)
            if error_mode in ["bar","both"]:
                plt.errorbar(
                    self.data_times, self.data_values,
                    yerr=self.data_errors,
                    fmt='none', ecolor='blue', elinewidth=0.7,
                    alpha=0.4, capsize=2, zorder=2
                )

            if error_mode in ["area","both"]:
                s_times, lower, upper = [], [], []
                i = 0
                while i < len(self.data_times)-6:
                    window_vals = self.data_values[i:i+5]
                    window_errs = self.data_errors[i:i+5]
                    lo = min(v - e for v, e in zip(window_vals, window_errs))
                    hi = max(v + e for v, e in zip(window_vals, window_errs))

                    s_times.append(self.data_times[i])
                    lower.append(lo)
                    upper.append(hi)
                    i += 5

                # tail
                window_vals = self.data_values[i:]
                window_errs = self.data_errors[i:]
                if window_vals:
                    lo = min(v - e for v, e in zip(window_vals, window_errs))
                    hi = max(v + e for v, e in zip(window_vals, window_errs))
                    s_times.append(self.data_times[i])
                    lower.append(lo)
                    upper.append(hi)

                plt.fill_between(s_times, lower, upper, color='blue', alpha=0.2, zorder=1)

            w = 0.02

            if self.noise_range != None:
                plt.axhline(y=self.noise_range[0], color='gray', linestyle='--')
                plt.axhline(y=self.noise_range[1], color='gray', linestyle='--')
                w = (self.noise_range[1]-1)*2

            
            if self.discontinuities != None:
                d_array = []
                for disc in self.discontinuities:
                    if "line" in kwargs.get("disc_mode", "bar"):
                        plt.axvline(x=disc["from"], color='red', linestyle='-')

                    # Format: tuple (start, length)
                    if "bar" in kwargs.get("disc_mode", "bar"):
                        d_array.append((disc["from"],disc["to"]-disc["from"]))
                
                plt.broken_barh(d_array, (1, w), align="center", color=(1,0,0,0.3))
            
            
            
            if kwargs.get("transit_mask", False) and self.data_masked:
                # Plot the masked areas saved in self.mask_array using a horizontal bar
                mask_array = self.mask_array
                if mask_array is not None:
                    mask_intervals = []
                    in_mask = False
                    start_time = None

                    for i, masked in enumerate(mask_array):
                        if masked and not in_mask:
                            in_mask = True
                            start_time = self.data_times[i]
                        elif not masked and in_mask:
                            in_mask = False
                            end_time = self.data_times[i]
                            mask_intervals.append((start_time, end_time - start_time))

                    if in_mask:
                        end_time = self.data_times[-1]
                        mask_intervals.append((start_time, end_time - start_time))

                    plt.broken_barh(mask_intervals, (1, w), color=(0.5, 0.5, 0.5, 0.3), align="center")

            if kwargs.get("transit_models", False) and self.transits:
                for i, t in enumerate(self.transits):
                    if not t.deep_analyzed:
                        try: t.deep_analyze()
                        except (FittingFailed, NotAnalyzed): continue
                    if t.deep_analyzed:
                        mt, mf = t.model_curve()
                        plt.plot(mt, mf, color="red", lw=2,
                                label="Transit model" if i == 0 else None, zorder=5)
                plt.legend()

            if self.transits != None:
                t_array = []
                i = 0
                avg_y = (self.noise_range[0]+self.noise_range[1])/2
                for t in self.transits:
                    s = t.start
                    e = t.end
                    t_array.append((s,e-s))
                    plt.text(s,self.noise_range[1]+0.005,f"T#{i}\nD: {(t.depth*100):.4}%")
                    plt.broken_barh([(s,e-s)], (avg_y - t.depth,0.0001), color='green', align="center")
                    i += 1
                plt.broken_barh(t_array, (1, 0.1), color=(0.3,0.3,0.5,0.3), align="center")
            

        plt.show()

    def export_plot(self,annotated:bool=False):
        pass

    def show_periodogram(self, annotated:bool=False):
        """
        Only available when data was freshly loaded from satellites.
        """
        p = self.periodogram
        p.plot()
        plt.show()

    # Various values
    def start(self):
        for t in self.data_times:
            if t > 10:
                return t
        return 0

    
    def end(self):
        for t in self.data_times[-1:]:
            if t > 10:
                return t
        return time()

    
    # Debug/Telemetry
    def start_stopwatch(self):
        self.stopwatch_start = time()

    def stop_stopwatch(self,label):
        t = time()-self.stopwatch_start
        self.debug_times[label]=t
        return time()-self.stopwatch_start
        



# Important class that contains all the measurements and results and can store them.
class star_observation:
    target = ""
    MEASURMENT_RESULT_DIRECTORY = "./error"
    measurements = []
    good_measurements = []
    log_debug = False
    log_warnings = True
    log_errors = True
    debug_times = {}
    merged_data_pairs = {}

    stopwatch_starts = {}
    
    progress = None

    calculated = False

    max_t_depth = "N/A"
    t_depth = "N/A"
    t_duration = "N/A"
    avg_period = "N/A"

    star_radius = 0*u.m
    planet_radius = 0*u.m

    # INIT
    def __init__(self, star_name, depth = -1, debug = False):
        self.target = star_name
        self.measurement_depth = depth
        self.MEASURMENT_RESULT_DIRECTORY = f"measurements/{star_name}/"
        self.log_debug = debug
        
    
    def set_log_params(self, debug, warn, error):
        self.log_debug = debug
        self.log_warnings = warn
        self.log_errors = error
    
    def _load_cache(self):
        try:
            with open(CACHE_FILE_PATH, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            self.log_err("Cache file doesn't exist")
        except json.JSONDecodeError:
            self.log_err("Failed to parse cache file")
        return {}

    def get_star_data(self, disallow_cache_read: bool = False, disallow_cache_write: bool = False):
        cache = self._load_cache()
        entry = cache.get("star_data", {}).get(self.target)

        if entry and "names" in entry and not disallow_cache_read:
            self.log_dbg("Found star data in cache")
        else:
            self.log_dbg("Querying TIC")
            res = Catalogs.query_object(self.target, catalog="TIC")
            row = res[np.argmin(res["dstArcSec"])]

            rad = clean(row["rad"])  # solar radii
            name_cols = {"TIC": "ID", "HIP": "HIP", "TYC": "TYC", "Gaia": "GAIA",
                        "2MASS": "TWOMASS", "KIC": "KIC", "UCAC4": "UCAC",
                        "AllWISE": "ALLWISE", "SDSS": "SDSS"}
            names = [f"{p} {clean(row[c])}" for p, c in name_cols.items()
                    if c in row.colnames and clean(row[c])]

            entry = {
                "radius": rad * SOLAR_RADIUS.to(u.m).value if rad else None,  # m
                "mass": clean(row["mass"]),            # solar masses
                "tmag": clean(row["Tmag"]),            # TESS magnitude
                "vmag": clean(row["Vmag"]),            # V magnitude
                "teff": clean(row["Teff"]),            # K
                "obj_type": clean(row["objType"]),     # STAR / EXTENDED
                "lum_class": clean(row["lumclass"]),   # DWARF / GIANT
                "names": names,
            }

            if not disallow_cache_write:
                cache.setdefault("star_data", {})[self.target] = entry
                with open(CACHE_FILE_PATH, "w") as f:
                    json.dump(cache, f, indent=2)

        self.star_data = entry
        self.star_radius = entry["radius"] * u.m if entry["radius"] else None
        return self.star_radius
        
        
    
    def observe(self) -> None:
        """
        Starts the observation

        :return: None
        """
        self.log_dbg("Getting star data...")
        self.get_star_data()
        self.log_dbg("Successfully loaded star data")
        self.progress = Progress()
        self.progress.start()
        obs_prog = self.progress.add_task(f"Observing [{self.target}]",total=100)
        try:
            # Gets the target amount measurements from the satellites (specified in self.measurement_depth)
            self.measurements = []
            self.log_dbg(f"Observing with a depth of {self.measurement_depth} measurements")
            self.log_dbg(f"  Searching pixelfiles...")
            self.start_stopwatch()
            self.start_stopwatch(1)
            search = lk.search_targetpixelfile(self.target)
            self.log_dbg(f"  Search complete, time elapsed: {self.stop_stopwatch(label='Search time')}s")
            self.progress.update(obs_prog, advance=20)
            s = 80/self.measurement_depth
            for current_index in range(self.measurement_depth):
                self.log_dbg(f"  Getting measurement #{current_index}")
                current_measurement = None
                try:
                    current_measurement = measurement(self.target, current_index, search)
                    self.log_dbg(f"  ... Created object")
                    current_measurement.load_data_from_sat()
                except IndexError:
                    self.log_wrn(f"[!] No more data to load, stopping iteration.")
                    if current_index == 0:
                        self.log_err(f"[!!] No data!")
                        raise NoDataFound(f"No data found for target {self.target}. Check for typos and if it even exists. Note: Pass STAR NAME as argument not planet name!")
                    break
                
                self.progress.update(obs_prog, advance=s/3)
                self.log_dbg(f"  ... Loaded data")
                current_measurement.parse()
                self.log_dbg(f"  ... Parsed data")
                self.progress.update(obs_prog, advance=s/3)
                current_measurement.analyze()
                self.log_dbg(f"  ... Analyzed data")
                self.progress.update(obs_prog, advance=s/3)
                self.measurements.append(current_measurement)
                self.log_dbg(f"  ... Stored data")
        except Exception as e:
            self.progress.update(obs_prog,visible=False)
            self.progress.stop()
            raise(e)
            
        self.log_dbg(f"Finished observing {self.target}. ")
        self.progress.update(obs_prog, completed=100, visible=False)
        self.progress.stop()
    
    def check_data(self) -> None:
        """
        Checks each measurement and removes those with a highly unusual period time.

        :return: None
        """
        self.good_measurements = []
        if len(self.measurements) == 0: return
        allps = []
        for m in self.measurements:
            if len(m.transits) > 3:
                allps.append(m.period)
        allps = remove_outliers(allps,int(len(allps)*0.3))
        if len(allps) == 0: return
        avg_period = sum(allps)/len(allps)

        for m in self.measurements:
            delta = abs(avg_period-m.period)
            if delta < 0.5*u.day:
                self.good_measurements.append(m)

    def analyze_transits(self, deep_analyze:bool=False) -> None:
        """
        Starts Transit analysis. Collects all the transit data available, removes outliers and saves the mean depth and duration.
        
        :param deep_analyze: If True, will also perform a deep analysis of each transit to get more accurate results. This is very slow and not recommended for large datasets.
        :return: None
        """
        # Check if there's enough data
        if len(self.good_measurements) == 0: return

        # Planet period analysis
        allps = []
        for m in self.good_measurements:
            allps.append(m.period)
        allps = remove_outliers(allps,int(len(allps)*0.3))
        if len(allps) == 0: return
        self.avg_period = sum(allps)/len(allps)



        ts = []
        for m in self.good_measurements:
            ts += m.transits
        if len(ts) == 0: return

        # Transit depth analysis
        # Transit duration analysis

        depths = []
        durations = []
        for t in ts:
            if deep_analyze:
                if not (t.deep_analyzed and t.analyzed): t.analyze(deep_analyze=True)
            else:
                if not t.analyzed: t.analyze()

            depths.append(t.depth)
            durations.append(t.duration_s)
        
        depths = remove_outliers(depths,int(len(depths)*0.3))
        depths.sort()
        depths.reverse()
        self.max_t_depth = depths[0]
        depths = depths[:int(len(depths)*0.5)]
        if len(depths) != 0:
            self.t_depth = sum(depths)/len(depths)
        else:
            self.t_depth = 0

        durations = remove_outliers(durations, int(len(durations)*0.3))
        if len(durations) == 0:
            self.t_duration = 0
        else:
            self.t_duration = sum(durations)/len(durations)

        self.planet_radius = math.sqrt(self.t_depth) * self.star_radius 


        if self.star_data and self.star_data.get("mass"):
            star_mass = self.star_data["mass"] * u.M_sun
            semi_major_axis = ((self.avg_period**2 * G * star_mass) / (4 * np.pi**2))**(1/3)
            self.semi_major_axis = semi_major_axis.to(u.m)
        else:
            self.semi_major_axis = "N/A"


        self.calculated = True


    def merge_data(self) -> dict:
        """
        Combines all measurement points/lightcurves into one big mess.
        Stores the data to self.merged_data_pairs and returns it
        Not recommended but do as you feel!

        :return: Dict: key is UTC time as string and value is float
        """
        self.log_dbg("Merging data points from all measurements")
        self.start_stopwatch()
        self.merged_data_pairs = {}
        for merging_measurement in self.measurements:
            data_pairs = merging_measurement.data_pairs
            for data_time in data_pairs.keys():
                data_value = data_pairs[data_time]
                self.merged_data_pairs[str(data_time)]=data_value
        self.log_dbg(f"Merging complete, duration: {self.stop_stopwatch(label='Data merging')}s")
        self.stop_stopwatch(index=1,label="Load to parse time")
        return self.merged_data_pairs

            
        

    # EXPORT
    # def export_measurement_as_csv(self,measurement_index,output_location_override: str = ""):
    #     self.start_stopwatch(index=4)
    
    def save(self) -> dict:
        sd = {
            "type": "star_observation",
            "target": self.target,
            "MEASUREMENT_RESULT_DIRECTORY": self.MEASURMENT_RESULT_DIRECTORY,
            "log_debug": self.log_debug,
            "log_warnings": self.log_warnings,
            "log_errors": self.log_errors,
            "debug_times": self.debug_times,
            "stopwatch_starts": self.stopwatch_starts,
            "calculated": self.calculated,
            "max_t_depth": self.max_t_depth,
            "t_depth": self.t_depth,
            "t_duration": self.t_duration,
            "avg_period": float(self.avg_period.value) if str(self.avg_period)!="N/A" else "N/A", 
            "star_radius": float(self.star_radius.value), # https://docs.astropy.org/en/stable/units/quantity.html
            "planet_radius": float(self.planet_radius.value if str(self.planet_radius)!="N/A" else "N/A"),
            "measurements": []
        }
        for m in self.measurements:
            sd["measurements"].append(m.save())
        return sd

    def load(self,load_dict):
        self.stopwatch_starts = load_dict.get("stopwatch_starts",{})
        self.start_stopwatch(5)

        self.log_dbg("Loading from dict")
        ld = load_dict
        if load_dict["type"] != "star_observation": return TypeError("The provided dict is not for a star observation")
        self.target = ld["target"]
        self.MEASURMENT_RESULT_DIRECTORY = ld["MEASUREMENT_RESULT_DIRECTORY"]
        self.log_debug = ld["log_debug"]
        self.log_warnings = ld["log_warnings"]
        self.log_errors = ld["log_errors"]
        self.debug_times = ld["debug_times"]
        self.calculated = ld["calculated"]
        self.max_t_depth = ld["max_t_depth"]
        self.t_depth = ld["t_depth"]
        self.t_duration = ld["t_duration"]
        self.star_radius = ld.get("star_radius",-1) * u.m
        self.planet_radius = ld.get("planet_radius",-1) * u.m 
        self.log_dbg("Loaded statics")
        self.avg_period = ld["avg_period"] * u.day # https://docs.astropy.org/en/stable/units/quantity.html
        self.log_dbg("Loaded period")

        self.start_stopwatch(0)
        self.log_dbg("Getting search...")
        self.search = lk.search_targetpixelfile(self.target)
        self.log_dbg(f"Got search in {self.stop_stopwatch('Search from dict')}s")


        self.log_dbg(f"Getting {len(ld['measurements'])} measurements")
        self.measurements = []
        i = 0
        for m in ld["measurements"]:
            self.start_stopwatch()
            new_instance = measurement(self.target,m["index"],self.search)
            new_instance.load(m)
            self.measurements.append(new_instance)

            self.log_dbg(f"  Loaded measurement #{i} in {self.stop_stopwatch(f'Load measurement {i} from dict'):.3}s    ({i+1}/{len(ld['measurements'])})")
            i += 1
        self.log_dbg("Finished loading measurements")
        self.log_dbg(f"Loaded from dict in {self.stop_stopwatch('Load from dict',index=5)}")



    def export_merged_as_csv(self,output_location_override: str=""):
        self.start_stopwatch()
        self.log_dbg(f"Started CSV export of merged data")
        csv_string = "Time (unix timestamp),Relative flux,Error"

        data = self.merged_data_pairs
        for key in data.keys():
            value = data[key][0]
            error = data[key][1]
            csv_string+=f"\n{key},{value},{error}"
        
        self.log_dbg(f"  CSV string generation completed after {self.stop_stopwatch('CSV string generation')}s")
        self.start_stopwatch(2)

        if output_location_override == "":
            output_location_override = self.MEASURMENT_RESULT_DIRECTORY+f"measurement_{self.target}.csv"
        
        # Generate missing directories (so no Errno)
        output_dir = os.path.dirname(output_location_override)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        self.log_dbg(f"  Directories created after {self.stop_stopwatch('CSV export dir gen',index=2)}s")

        # Write to the file
        with open(output_location_override, "w", encoding="utf-8") as csv_file:
            csv_file.write(csv_string)
        
        self.log_dbg(f"  CSV stored in {self.stop_stopwatch('CSV export')}s")

            
    def export_overview(self,path):
        open(path, "w").write(self.get_overview())

    # FANCY DISPLAYING
    def print_overview(self):
        print(self.get_overview())

    def get_overview(self):
        # Returns a simple overview of measurements and calculations as a string
        lines = []
        lines.append(f"[ OBSERVATION STATUS FOR TARGET {self.target} ]")
        lines.append(f"|  Parameters:")
        lines.append(f"|   Depth:                              {self.measurement_depth}")
        lines.append(f"|   Result directory:                   {self.MEASURMENT_RESULT_DIRECTORY}")
        lines.append(f"|   Debug:                              {self.log_debug}")
        lines.append(f"[ ============================================================= ]")
        lines.append(f"|  System data:")
        lines.append(f"|   Target star:                        {self.target}:")
        lines.append(f"|   Star radius:                        {self.star_radius}")
        lines.append(f"|   Planet radius:                      {self.planet_radius}")
        lines.append(f"[ ============================================================= ]")
        lines.append(f"|  Measurement results")
        lines.append(f"|   Loaded measurement data files:      {len(self.measurements)}/{self.measurement_depth}")

        for meas in self.measurements:
            prefix = Fore.YELLOW if meas not in self.good_measurements else ""
            suffix = Fore.RESET if prefix else ""

            lines.append(f"{prefix}|   @ Measurement # {meas.index}")
            lines.append(f"|     Data from:                        {meas.get_data_source()}")
            lines.append(f"|     Start:                            {meas.start()}")
            lines.append(f"|     End:                              {meas.end()}")
            lines.append(f"|     AVG scan interval:                {meas.get_avg_scan_interval()}")
            lines.append(f"|     Loaded:                           {meas.is_loaded}")
            lines.append(f"|     Parsed:                           {meas.is_loaded}")
            lines.append(f"|     Analyzed:                         {meas.is_analyzed}")
            lines.append(f"|     Period:                           {meas.period}")
            lines.append(f"|     # of slices:                      {len(meas.data_values)}")

            if meas.discontinuities is not None:
                lines.append(f"|     # of discontinuities:             {len(meas.discontinuities)}")
            else:
                lines.append(f"|     # of discontinuities:             N/A")

            lines.append(f"|     ")
            if meas.transits is not None:
                lines.append(f"|     # of transits found:              {len(meas.transits)}")
                lines.append(f"|     Transits:")
                for t in meas.transits:
                    lines.append(f"|       {t.showshort()}")
            else:
                lines.append(f"|     # of transits found:              N/A")

            lines.append(f"|     ")
            lines.append(f"|     Times:")

            times = meas.debug_times
            total = 0
            for k, v in times.items():
                total += v
                lines.append(f"|      {k}: \t\t{round(v*1000000)/1000}ms")
            lines.append(f"|      Total: \t\t\t{round(total*1000000)/1000}ms")

            lines.append(f"{suffix}", )  # closes color if opened

        lines.append(f"[ ============================================================= ]")

        if self.calculated:
            lines.append(f"|  Calculation results")
            lines.append(f"|    Max. transit depth:         {self.max_t_depth*100:.10}% / {pct_to_mmag(self.max_t_depth*100):.10}")
            lines.append(f"|    Avg. transit depth:         {self.t_depth*100:.10}% / {pct_to_mmag(self.t_depth*100):.10}")
            lines.append(f"|    Avg. transit duration:      {self.t_duration/60/60:.10}h / {self.t_duration/60} min")
            lines.append(f"|    Avg. period:                {self.avg_period:.10}")
            lines.append(f"|    Planet radius:              {self.planet_radius:.10}")
            lines.append(f"|    Planet radius in R_J:       {(self.planet_radius/JUPITER_RADIUS):.10}")
            lines.append(f"|    Semi-major axis:            {self.semi_major_axis:.10}")
            lines.append(f"[ ============================================================= ]")

        lines.append(f"|  Stopwatch times")
        times = self.debug_times
        total = 0
        for k, v in times.items():
            total += v
            lines.append(f"|    {k}:\t\t\t\t{round(v*1000000)/1000}ms")

        return "\n".join(lines)



    def show_transit_plot(self):
        # Shows a plot of a found transit
        self.log_wrn("'show_transit_plot' is not implemented")
        pass

    def show_labeled_transit_plot(self):
        # Shows a labeled plot of a found transit, including information like start of egress or ingress.
        self.log_wrn("'show_labeled_transit_plot' is not implemented")
        pass
    

    # HELPER FUNCTIONS
    def log_dbg(self,text):
        if self.log_debug: print(f"{datetime.now().strftime('%d/%m/%Y - %H:%M:%S')} | [DBG]({self.target}) | {text}")

    def log_wrn(self,text):
        if self.log_warnings: print(f"{Fore.YELLOW}{datetime.now().strftime('%d/%m/%Y - %H:%M:%S')} | [WRN]({self.target}) | {text}{Fore.RESET}")

    def log_err(self,text):
        if self.log_errors: print(f"{Fore.RED}{datetime.now().strftime('%d/%m/%Y - %H:%M:%S')} | [ERR]({self.target}) | {text}{Fore.RESET}")

    def start_stopwatch(self,index=0):
        self.stopwatch_starts[str(index)] = time()

    def stop_stopwatch(self,label=None,index=0):
        t = time()-self.stopwatch_starts[str(index)]
        if label == None:
            label = f"Measurement {len(self.debug_times)}"
        self.debug_times[label]=t
        return time()-self.stopwatch_starts[str(index)]



print(Fore.BLUE+"Hello from planetlib "+str(VERSION)+"!"+Fore.RESET)