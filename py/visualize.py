import time
import sys
sys.path.append("/home/tbent/packages/visit2_7_0/2.7.0/linux-x86_64/lib/site-packages/")
import visit
visit.Launch('/home/tbent/packages/visit2_7_0/bin/')
filename_prefix = 'solution_'
filename_prefix = 'init_refinement_'
for i in range(12):
    time.sleep(0.5)
    visit.DeleteAllPlots()
    visit.OpenDatabase('./data/test/' + filename_prefix + str(i) + '.0000.pvtu')
    visit.DefineScalarExpression('logabsoldszx', 'log(abs(old_szx))')
    # visit.AddPlot('Pseudocolor', 'logabsoldszx')
    visit.AddPlot('Pseudocolor', 'vel')
    # visit.AddPlot('Mesh', 'mesh')
    visit.DrawPlots()

