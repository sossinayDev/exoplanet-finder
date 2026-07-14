from json import dumps, loads

from planetlib import star_observation, measurement

obs_tres = star_observation("TOI 2046", 1, True)
obs_tres.load(loads(open("test.json","r",encoding="utf-8").read()))
# obs_tres.observe()
# obs_tres.print_overview()
# obs_tres.analyze_transits()
obs_tres.print_overview()
m1: measurement = obs_tres.measurements[0]
m1.show_plot(annotated = True)

# open("test.json", "w", encoding="utf-8").write(dumps(obs_tres.save()))