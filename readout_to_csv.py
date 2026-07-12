import os

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("QtAgg")
import lightkurve as lk
from datetime import datetime
from colorama import Fore
from time import time

# Custom exceptions (Errors) Source: https://www.geeksforgeeks.org/python/define-custom-exceptions-in-python/
class NoDataFound:
    # Called when parsing satellite data was not found (! NOT WHEN INDEX TO HIGH !)
    def __init__(self, message = "No data was found"):
        self.message = message
        super().__init__(self.message)
    
    def __str__(self):
        return f"{self.message}"


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

    data_values = []
    data_times = []
    data_pairs = {}

    stopwatch_start = 0
    debug_times = {}

    def __init__(self, star_name, measurement_index, search, autoload: bool = False):
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
        self.curve = self.pixelfile.to_lightcurve(method="pld").remove_outliers().flatten()
        # Convert it to a periodogram
        self.period = self.curve.to_periodogram("bls").period_at_max_power
        self.curve.fold(self.period).scatter();
        # self.filled_curve = self.curve.fill_gaps() # Experimental, check results
        self.filled_curve = self.curve
        self.stop_stopwatch("Enhancing duration")

        
        self.is_loaded = True # Save progress
        if autoparse: self.parse()

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
            v = float(temp_value_list[scanning_index].value)

            self.data_times.append(t)
            self.data_values.append(v)
            self.data_pairs[t] = v
        
        self.is_parsed = True # Save progress
        self.stop_stopwatch("Parsing time")
    
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
    my_measurements = []
    log_debug = False
    log_warnings = True
    log_errors = True
    debug_times = {}
    merged_data_pairs = {}

    stopwatch_starts = {}

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
        # Gets the desired amount measurements from the satellites (specified in self.measurement_depth)
        self.my_measurements = []
        self.log_dbg(f"Observing with a depth of {self.measurement_depth} measurements")
        self.log_dbg(f"  Searching pixelfiles...")
        self.start_stopwatch()
        self.start_stopwatch(1)
        search = lk.search_targetpixelfile(self.target)
        self.log_dbg(f"  Search complete, time elapsed: {time()-self.stop_stopwatch(label='Search time')}s")
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
            self.log_dbg(f"  ... Loaded data")
            current_measurement.parse()
            self.log_dbg(f"  ... Parsed data")
            self.my_measurements.append(current_measurement)
            self.log_dbg(f"  ... Stored data")
        self.log_dbg(f"Finished observing {self.target}. ")
    
    def merge_data(self):
        self.log_dbg("Merging data points from all measurements")
        self.start_stopwatch()
        self.merged_data_pairs = {}
        for merging_measurement in self.my_measurements:
            data_pairs = merging_measurement.data_pairs
            for data_time in data_pairs.keys():
                data_value = data_pairs[data_time]
                self.merged_data_pairs[str(data_time)]=data_value
        self.log_dbg(f"Merging complete, duration: {self.stop_stopwatch(label='Data merging')}s")
        self.stop_stopwatch(index=1,label="Load to parse time")

            
        

    # EXPORT
    def export_as_csv(self,output_location_override: str=""):
        csv_string = "Time (unix timestamp),Flux (e-/s)"

        data = self.merged_data_pairs
        for key in data.keys():
            value = data[key]
            csv_string+=f"\n{key},{value}"
        
        if output_location_override == "":
            output_location_override = self.MEASURMENT_RESULT_DIRECTORY+f"measurement_{self.target}.csv"


        # Generate missing directories (so no Errno)
        output_dir = os.path.dirname(output_location_override)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Write to the file
        with open(output_location_override, "w", encoding="utf-8") as csv_file:
            csv_file.write(csv_string)
        

            
            

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
        print(f"|   Loaded measurement data files:      {len(self.my_measurements)}/{self.measurement_depth}")

        for meas in self.my_measurements:
            print(f"|   @ Measurement # {meas.index}")

            print(f"|     Start:            {meas.data_times[0]}")
            print(f"|     End:              {meas.data_times[-1]}")

            print(f"|     Loaded:           {meas.is_loaded}")
            print(f"|     Parsed:           {meas.is_loaded}")
            print(f"|     Period:           {meas.period}")
            print(f"|     # of slices:      {len(meas.data_values)}")
            print(f"|     Times:")

            times = meas.debug_times
            total = 0
            for k in times.keys():
                v = times[k]
                total += v
                print(f"|      {k}: 		{round(v*1000000)/1000}ms")
            print(f"|      Total: 			{round(total*1000000)/1000}ms")
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



obs_tres = star_observation("TrES-3 b", 3, True)
obs_tres.observe()
obs_tres.merge_data()
obs_tres.print_overview()
obs_tres.export_as_csv()