from json import dumps, loads

from planetlib import star_observation, measurement

obs_tres = star_observation("KELT-23Ab", 10)
# obs_tres.load(loads(open("measurements/kelt23ab_4.json.json","r",encoding="utf-8").read()))

obs_tres.observe()
obs_tres.check_data()
obs_tres.analyze_transits()
obs_tres.print_overview()

open("measurements/kelt23ab_4.json", "w", encoding="utf-8").write(dumps(obs_tres.save()))


for m in obs_tres.measurements:
    m.show_plot(annotated = True)