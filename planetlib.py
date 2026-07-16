import math
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("QtAgg")
import lightkurve as lk
from datetime import datetime
from colorama import Fore
import time
from time import time
import astropy.units as u
from rich.progress import Progress # Q: https://www.geeksforgeeks.org/python/progress-bars-in-python/

def pct_to_mmag(depth_pct):
    depth_pct = 1.60277
    flux_ratio = 1 - depth_pct/100

    delta_mag = -2.5 * np.log10(flux_ratio) * u.mag
    return delta_mag.to(u.mmag)

def strfunix(unix,string:str="%d/%m/%Y - %H:%M:%S"):
    return datetime.fromtimestamp(unix).strftime(string)

def remove_outliers(l,n=1):
    l = l.copy()
    for e in range(n):
        l.remove(max(l))
        l.remove(min(l))
    return l
    
def ao(num,arr,pos,inv:bool=False):
    sum = 0
    for j in range(num):
        if inv:
            sum += arr[pos-j]
        else:
            sum += arr[pos+j]
    return sum/num

# Custom exceptions (Errors) Source: https://www.geeksforgeeks.org/python/define-custom-exceptions-in-python/
class NoDataFound:
    # Called when parsing satellite data was not found (! NOT WHEN INDEX TO HIGH !)
    def __init__(self, message = "No data was found"):
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return f"{self.message}"

class transit:
    times = []
    values = []

    start = 0
    end = 0
    depth = 0
    noise_range = []
    analyzed = False

    duration_s = 0

    def __init__(self,start,end,times,values,noiserange):
        self.noise_range = noiserange

        self.times = times
        self.values = values

        self.start = start
        self.end = end

        self.start_index = self.times.index(self.start)
        self.end_index = self.times.index(self.end)

        self.analyze()
    
    def analyze(self):
        data_range = self.values[self.start_index:self.end_index]
        data_range.sort()
        
        lowest_quarter = data_range[:int(len(data_range)*5/100)]
        average_minimum_value = data_range[0]
        if len(lowest_quarter) > 2:
            average_minimum_value = sum(lowest_quarter)/len(lowest_quarter) # Average of the lowest 2% of vals
            

        self.depth = (self.noise_range[0]+self.noise_range[1])/2-average_minimum_value

        self.duration_s = self.end - self.start
        self.analyzed = True

    
    def save(self):
        return {
            "start": self.start,
            "start_index": self.start_index,
            "end": self.end,
            "end_index": self.end_index,
            "depth": self.depth,
            "duration_s": self.duration_s,
            "analyzed": self.analyzed
        }
        

    def showshort(self):
        return f"START: {strfunix(self.start)} | END: {strfunix(self.end)} | Depth: {self.depth}"
        

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

    data_values = []
    data_times = []
    data_pairs = {}

    transits = None

    stopwatch_start = 0
    debug_times = {}

    is_analyzed = False

    discontinuities = None

    def __init__(self, star_name, measurement_index, search, autoload: bool = False):
        self.data_values = []
        self.data_times = []
        self.data_pairs = {}
        self.target = star_name
        self.search = search
        self.index = measurement_index
        if autoload:
            self.load_data_from_sat(autoparse=True)
    
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
        self.curve = self.pixelfile.to_lightcurve(aperture_mask=self.pixelfile.pipeline_mask).remove_outliers(sigma=6).flatten()
        # self.curve = self.pixelfile.to_lightcurve(method="pld").remove_outliers().flatten()
        # Convert it to a periodogram
        self.period = self.curve.to_periodogram("bls").period_at_max_power
        self.curve.fold(self.period).bin(10).remove_nans();
        # self.filled_curve = self.curve.fill_gaps() # Experimental, check results
        self.filled_curve = self.curve
        self.stop_stopwatch("Enhancing duration")
        
        self.is_loaded = True # Save progress
        if autoparse: self.parse()

    def load(self,ld):
        self.period = ld["period_d"] * u.day
        self.is_loaded = ld["is_loaded"]
        self.is_parsed = ld["is_parsed"]
        self.is_analyzed = ld["is_analyzed"]
        self.noise_range = (ld["noise_range"][0], ld["noise_range"][1])
        self.data_values = ld["data_values"]
        self.data_times = ld["data_times"]
        self.debug_times = ld["debug_times"]
        self.stopwatch_start = ld["stopwatch_start"]
        self.discontinuities = ld["discontinuities"]
        self.transits = []

        for t in ld["transits"]:
            new_inst = transit(t["start"],t["end"],self.data_times,self.data_values,self.noise_range)
            new_inst.depth = t["depth"]
            new_inst.duration_s = t["duration_s"]
            new_inst.analyzed = t["analyzed"]
            self.transits.append(new_inst)

    def parse(self):
        self.start_stopwatch()
        temp_time_list = list(self.filled_curve.time)
        temp_value_list = list(self.filled_curve.flux)

        self.data_values = []
        self.data_times = []
        self.data_pairs = {}


        # Iterate through Quantity objects and turn them into something I understand
        for scanning_index in range(len(temp_time_list)):
            t = temp_time_list[scanning_index].to_value("unix")
            if t < 10:
                continue
            v = float(temp_value_list[scanning_index].value)

            self.data_times.append(t)
            self.data_values.append(v)
            self.data_pairs[t] = v
        
        self.is_parsed = True # Save progress
        self.stop_stopwatch("Parsing time")

    # Analysis
    def analyze(self):
        self.get_noise_range()
        self.find_discontinuities()
        self.find_transits()
        self.is_analyzed = True
    
    def get_avg_scan_interval(self):
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
                for j in range(-2,3):
                    if self.data_values[point_index+j] > n_min:
                        is_strong_negative = False
                        break
                if not is_strong_negative: continue
                v = current_value
                ci = point_index
                valid = True
                while v < n_max and ao(5,self.data_values,ci,inv=True)<epsilon and valid:
                    ci -= 1
                    if ci < 1:
                        valid = False
                    v = self.data_values[ci]
                transit_start = ci
                v = current_value
                ci = point_index
                while v < n_max and ao(5,self.data_values,ci)<epsilon and valid:
                    ci += 1
                    if ci > len(self.data_values)-1:
                        valid = False
                        break
                    v = self.data_values[ci]
                transit_end = ci

                if self.check_for_discontinuity((transit_start,transit_end)): valid = False

                if valid:
                    transits.append(transit(self.data_times[transit_start],self.data_times[transit_end],self.data_times,self.data_values,self.noise_range))
                    added_until = transit_end
        
        self.transits = transits



    
    # Export/Loading
    def save(self):
        sd = {
            "target": self.target,
            "index": self.index,
            "period_d": float(str(self.period).split(" ")[0]),
            "is_loaded": self.is_loaded,
            "is_parsed": self.is_parsed,
            "is_analyzed": self.is_analyzed,
            "noise_range": self.noise_range,
            "data_values": self.data_values,
            "data_times": self.data_times,
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

        csv_string = "Time (unix timestamp),Relative flux"

        data = self.data_pairs
        for key in data.keys():
            value = data[key]
            csv_string+=f"\n{key},{value}"

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
            plt.scatter(self.data_times,self.data_values)
        plt.xlabel("Time (unix timestamp)")
        plt.ylabel("Relative flux")
        plt.xlim(self.start(),self.end())
        plt.ylim(0,2)
        plt.title(f"Measurement #{self.index} ({self.target})")

        if kwargs.get("annotated", False):
            # Lines: https://matplotlib.org/stable/gallery/lines_bars_and_markers/axline.html
            # Broken horizontal bars: https://matplotlib.org/stable/gallery/lines_bars_and_markers/broken_barh.html
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
            
            
            if self.transits != None:
                t_array = []
                i = 0
                avg_y = (self.noise_range[0]+self.noise_range[1])/2
                for t in self.transits:
                    s = t.start
                    e = t.end
                    t_array.append((s,e-s))
                    plt.text(s,self.noise_range[1]+0.005,f"T#{i}\nD: {(t.depth*100):.4}%")
                    plt.broken_barh([(s,e-s)], (avg_y - t.depth,0.0001), color='green')
                    i += 1
                plt.broken_barh(t_array, (1, 0.1), align="center", color=(0.3,0.3,0.5,0.3))


        plt.show()
        pass

    def export_plot(self,annotated:bool=False):
        pass


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
    target = "TrES-3 b"
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

    t_depth = "N/A"
    t_duration = "N/A"
    avg_period = "N/A"

    # INIT
    def __init__(self, star_name, depth = -1, debug = False):
        self.target = star_name
        self.measurement_depth = depth
        self.MEASURMENT_RESULT_DIRECTORY = f"/home/yanis/Nextcloud/000_School/Maturaarbeit/code/exoplanet-finder/measurements/{star_name}/"
        self.log_debug = debug
        
    
    def set_log_params(self, debug, warn, error):
        self.log_debug = debug
        self.log_warnings = warn
        self.log_errors = error
    
    def observe(self):
        self.progress = Progress()
        self.progress.start()
        obs_prog = self.progress.add_task("Observing...",total=100)
        # Gets the desired amount measurements from the satellites (specified in self.measurement_depth)
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
            except IndexError:
                self.log_wrn(f"[!] No more data to load, stopping iteration.")
                if current_index == 0:
                    self.log_err(f"[!!] ")
                    raise NoDataFound(f"No data found for target {self.target}. Check for typos and if it even exists. Note: Pass STAR NAME as argument not planet name!")
                break
            self.log_dbg(f"  ... Created object")
            current_measurement.load_data_from_sat()
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
        self.log_dbg(f"Finished observing {self.target}. ")
        self.progress.update(obs_prog, completed=100)
        self.progress.stop()
    
    def check_data(self):
        if len(self.measurements) == 0: return
        # Remove measurements with high period delta
        allps = []
        for m in self.measurements:
            allps.append(m.period)
        allps = remove_outliers(allps,int(len(allps)*0.3))
        avg_period = sum(allps)/len(allps)

        for m in self.measurements:
            delta = abs(avg_period-m.period)
            if delta < 0.5*u.hour:
                self.good_measurements.append(m)

    def analyze_transits(self):
        # Check if there's enough data
        if len(self.good_measurements) == 0: return

        # Planet period analysis
        allps = []
        for m in self.good_measurements:
            allps.append(m.period)
        allps = remove_outliers(allps,int(len(allps)*0.3))
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
            if not t.analyzed: t.analyze()
            depths.append(t.depth)
            durations.append(t.duration_s)
        
        depths = remove_outliers(depths,int(len(depths)*0.3))
        self.t_depth = sum(depths)/len(depths)

        durations = remove_outliers(durations, int(len(durations)*0.3))
        self.t_duration = sum(durations)/len(durations)




        self.calculated = True


    def merge_data(self):
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

            
        

    # EXPORT
    def export_measurement_as_csv(self,measurement_index,output_location_override: str = ""):
        self.start_stopwatch(index=4)
    
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
            "t_depth": self.t_depth,
            "t_duration": self.t_duration,
            "avg_period": float(str(self.avg_period).split(" ")[0]),
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
        self.t_depth = ld["t_depth"]
        self.t_duration = ld["t_duration"]
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
        csv_string = "Time (unix timestamp),Relative flux"

        data = self.merged_data_pairs
        for key in data.keys():
            value = data[key]
            csv_string+=f"\n{key},{value}"
        
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

            
            

    # FANCY DISPLAYING
    def print_overview(self):
        # Prints a simple overview of measurements and calculations
        print(f"[ OBSERVATION STATUS FOR TARGET {self.target} ]")
        print(f"|  Parameters:")
        print(f"|   Depth:                              {self.measurement_depth}")
        print(f"|   Result directory:                   {self.MEASURMENT_RESULT_DIRECTORY}")
        print(f"|   Debug:                              {self.log_debug}")
        print(f"[ ============================================================= ]")
        print(f"|  Measurement results")
        print(f"|   Loaded measurement data files:      {len(self.measurements)}/{self.measurement_depth}")

        for meas in self.measurements:
            if not meas in self.good_measurements:
                print(Fore.YELLOW,end="")
            print(f"|   @ Measurement # {meas.index}")

            print(f"|     Start:                            {meas.start()}")
            print(f"|     End:                              {meas.end()}")
            print(f"|     AVG scan interval:                {meas.get_avg_scan_interval()}")

            print(f"|     Loaded:                           {meas.is_loaded}")
            print(f"|     Parsed:                           {meas.is_loaded}")
            print(f"|     Analyzed:                         {meas.is_analyzed}")

            print(f"|     Period:                           {meas.period}")
            print(f"|     # of slices:                      {len(meas.data_values)}")
            if meas.discontinuities != None:
                print(f"|     # of discontinuities:             {len(meas.discontinuities)}")
            else:
                print(f"|     # of discontinuities:             N/A")
            
            print(f"|     ")
            if meas.transits != None:
                print(f"|     # of transits found:              {len(meas.transits)}")
                print(f"|     Transits:")
                
                for t in meas.transits:
                    print(f"|       {t.showshort()}")
            else:
                print(f"|     # of transits found:              N/A")

            print(f"|     ")
            print(f"|     Times:")

            times = meas.debug_times
            total = 0
            for k in times.keys():
                v = times[k]
                total += v
                print(f"|      {k}: 		{round(v*1000000)/1000}ms")
            print(f"|      Total: 			{round(total*1000000)/1000}ms")

            print(Fore.RESET,end="")
        print(f"[ ============================================================= ]")
        if self.calculated:
            print(f"|  Calculation results")
            print(f"|    Avg. transit depth:    {self.t_depth*100:.5}% / {pct_to_mmag(self.t_depth*100):.5}")
            print(f"|    Avg. transit duration: {self.t_duration/60/60:.5}h / {self.t_duration/60} min")
            print(f"|    Avg. period:           {self.avg_period:.5}")

            print(f"[ ============================================================= ]")
        print(f"|  Stopwatch times")
        times = self.debug_times
        total = 0
        for k in times.keys():
            v = times[k]
            total += v
            print(f"|    {k}:				{round(v*1000000)/1000}ms")



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
        if self.log_debug: print(f"{datetime.now().strftime("%d/%m/%Y - %H:%M:%S")} | [DBG]({self.target}) | {text}")

    def log_wrn(self,text):
        if self.log_warnings: print(f"{Fore.YELLOW}{datetime.now().strftime("%d/%m/%Y - %H:%M:%S")} | [DBG]({self.target}) | {text}{Fore.RESET}")

    def log_err(self,text):
        if self.log_errors: print(f"{Fore.RED}{datetime.now().strftime("%d/%m/%Y - %H:%M:%S")} | [DBG]({self.target}) | {text}{Fore.RESET}")

    def start_stopwatch(self,index=0):
        self.stopwatch_starts[str(index)] = time()

    def stop_stopwatch(self,label=None,index=0):
        t = time()-self.stopwatch_starts[str(index)]
        if label == None:
            label = f"Measurement {len(self.debug_times)}"
        self.debug_times[label]=t
        return time()-self.stopwatch_starts[str(index)]



