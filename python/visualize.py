import time
import sys
sys.path.append("/home/tbent/packages/visit2_7_0/2.7.0/linux-x86_64/lib/site-packages/")
import visit
visit.Launch('/home/tbent/packages/visit2_7_0/bin/')
for i in range(2, 3):
    time.sleep(0.5)
    visit.DeleteAllPlots()
    visit.OpenDatabase('./data/test/init_refinement_' + str(i) + '.0000.pvtu')
    visit.DefineScalarExpression('logabsoldszx', 'log(abs(old_szx))')
    visit.AddPlot('Pseudocolor', 'logabsoldszx')
    # visit.AddPlot('Pseudocolor', 'vel')
    # visit.AddPlot('Mesh', 'mesh')
    visit.DrawPlots()
